package me.xinzhang.spark.mllib

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.impurity.Entropy
import org.apache.spark.rdd.RDD


/**
  * Created by xin on 24/05/2017.
  */
object Classification {

  private val sc = new SparkContext("local[*]", "Classifications")

  Logger.getLogger("org").setLevel(Level.ERROR)


  def E0_LoadData(f: Double => Double): RDD[LabeledPoint] = {
    val rawData = sc.textFile("data/stumbleupon/train_noheader.tsv")
    val records = rawData.map(x => x.split("\t"))
    records.map{
      r =>
        val trimmed = r.map(_.replaceAll("\"", ""))
        val label = trimmed(r.size - 1).toInt
        val features = trimmed.slice(4, r.size - 1).map(d => if (d == "?") 0.0 else d.toDouble).map(f)
        LabeledPoint(label, Vectors.dense(features))
    }
  }

  import org.apache.spark.mllib.classification.ClassificationModel

  def E1_FourMethods() = {

    val data = E0_LoadData(d => d).cache()
    val nbData = E0_LoadData(d => if (d < 0) 0 else d).cache()

    val numIterations = 10
    val maxTreeDepth = 5

    val lrModel = LogisticRegressionWithSGD.train(data, numIterations)

    val svmModel = SVMWithSGD.train(data, numIterations)

    val nbModel = NaiveBayes.train(nbData)

    val dtModel = DecisionTree.train(data, Algo.Classification, Entropy, maxTreeDepth)


    def evaluation(data: RDD[LabeledPoint], model: ClassificationModel): Unit = {
      val total = data.map{x =>
        if (model.predict(x.features) == x.label) 1.0 else 0.0
      }.sum()
      val accuracy = total / data.count()
      println(model.getClass.getSimpleName + ": " + accuracy)
    }

    evaluation(data, lrModel)
    evaluation(data, svmModel)
    evaluation(nbData, nbModel)

    val dtTotalCorrect = data.map{
      x => val predicted = if (dtModel.predict(x.features) > 0.5) 1.0 else 0.0
        if (predicted == x.label) 1 else 0
    }.sum()
    val dtAccuracy = dtTotalCorrect / data.count()
    println(dtModel.getClass.getSimpleName + ": " + dtAccuracy)

  }

  import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

  def E2_ROCandAUC() = {
    val data = E0_LoadData(d => d).cache()
    val nbData = E0_LoadData(d => if (d < 0) 0.0 else d)
    val numIterations = 10
    val maxTreeDepth = 5

    val lrModel = LogisticRegressionWithSGD.train(data, numIterations)

    val svmModel = SVMWithSGD.train(data, numIterations)

    val nbModel = NaiveBayes.train(nbData)

    val dtModel = DecisionTree.train(data, Algo.Classification, Entropy, maxTreeDepth)


    val metrics = Seq(lrModel, svmModel).map{
      model =>
        val scoreAndLabels = data.map{
          point =>
            (model.predict(point.features), point.label)
        }
        val metrics = new BinaryClassificationMetrics(scoreAndLabels)
        (model.getClass.getSimpleName, metrics.areaUnderPR(), metrics.areaUnderROC())
    }

    val nbMetrics = Seq(nbModel).map{
      model =>
        val scoreAndLabels = nbData.map{
          point =>
            (model.predict(point.features), point.label)
        }
        val metrics = new BinaryClassificationMetrics(scoreAndLabels)
        (model.getClass.getSimpleName, metrics.areaUnderPR(), metrics.areaUnderROC())
    }

    val dtMetrics = Seq(dtModel).map{
      model =>
        val scoreAndLabels = data.map{
          point =>
            val score = model.predict(point.features)
            (if (score > 0.5) 1.0 else 0.0, point.label)
        }
        val metrics = new BinaryClassificationMetrics(scoreAndLabels)
        (model.getClass.getSimpleName, metrics.areaUnderPR(), metrics.areaUnderROC())
    }

    (metrics ++ nbMetrics ++ dtMetrics).foreach{
      case (m, pr, roc) =>
        println(f"$m, Area under PR: ${pr * 100}%2.4f%%, Area under ROC: ${roc * 100}%2.4f%%")
    }
  }

  import org.apache.spark.mllib.linalg.distributed.RowMatrix
  import org.apache.spark.mllib.feature.StandardScaler

  def E3_Scaling() = {
    val data = E0_LoadData(d => d).cache()
    val vectors = data.map(_.features)
    val matrix = new RowMatrix(vectors)
    val matrixSummary = matrix.computeColumnSummaryStatistics()
    println(matrixSummary.mean)
    println(matrixSummary.variance)

    val scaler = new StandardScaler(withMean = true, withStd = true).fit(vectors)
    val scaledData = data.map(lp => LabeledPoint(lp.label, scaler.transform(lp.features)))

    val numIterations = 10

    val lrModel = LogisticRegressionWithSGD.train(scaledData, numIterations)

    val svmModel = SVMWithSGD.train(scaledData, numIterations)

    Seq(lrModel, svmModel).map{
      model =>
        val scoreAndLabels = scaledData.map{
          point =>
            (model.predict(point.features), point.label)
        }
        val total = scoreAndLabels.map{ case (x, y) =>
          if (x == y) 1.0 else 0.0
        }.sum()
        val metrics = new BinaryClassificationMetrics(scoreAndLabels)
        (model.getClass.getSimpleName, total / scaledData.count(), metrics.areaUnderPR(), metrics.areaUnderROC())
    }.foreach{
      case (m, a, pr, roc) =>
        println(f"$m, Accuracy: ${a * 100}2.4f%%,Area under PR: ${pr * 100}%2.4f%%, Area under ROC: ${roc * 100}%2.4f%%")
    }
  }

  def E4_CategoryData() = {
    val rawData = sc.textFile("data/stumbleupon/train_noheader.tsv")
    val records = rawData.map(x => x.split("\t"))
    val categories = records.map(r => r(3)).distinct.collect.zipWithIndex.toMap
    println(categories)
    val numCategories = categories.size
    val dataNB = records.map{
      r =>
        val trimmed = r.map(_.replaceAll("\"", ""))
        val label = trimmed(r.size - 1).toInt
        val categoryIdx = categories(r(3))
        val categoryFeatures = Array.ofDim[Double](numCategories)
        LabeledPoint(label, Vectors.dense(categoryFeatures))
    }

    val nbModelCats = NaiveBayes.train(dataNB)

    Seq(nbModelCats).map{
      model =>
        val scoreAndLabels = dataNB.map{
          point =>
            (model.predict(point.features), point.label)
        }
        val total = scoreAndLabels.map{ case (x, y) =>
          if (x == y) 1.0 else 0.0
        }.sum()
        val metrics = new BinaryClassificationMetrics(scoreAndLabels)
        (model.getClass.getSimpleName, total / dataNB.count(), metrics.areaUnderPR(), metrics.areaUnderROC())
    }.foreach{
      case (m, a, pr, roc) =>
        println(f"$m, Accuracy: ${a * 100}2.4f%%,Area under PR: ${pr * 100}%2.4f%%, Area under ROC: ${roc * 100}%2.4f%%")
    }

  }

  import org.apache.spark.mllib.optimization.Updater
  import org.apache.spark.mllib.optimization.SimpleUpdater
  import org.apache.spark.mllib.optimization.L1Updater
  import org.apache.spark.mllib.optimization.SquaredL2Updater

  def E5_TuneParam() = {

    val data = E0_LoadData(d => d).cache()
    val vectors = data.map(_.features)
    val scaler = new StandardScaler(withMean = true, withStd = true).fit(vectors)
    val scaledData = data.map(lp => LabeledPoint(lp.label, scaler.transform(lp.features)))

    def trainWithParams(input: RDD[LabeledPoint], regParam: Double, numIteration: Int, updater: Updater, stepSize: Double) = {
      val lr = new LogisticRegressionWithSGD()
      lr.optimizer.setNumIterations(numIteration).setRegParam(regParam).setUpdater(updater).setStepSize(stepSize)
      lr.run(input)
    }

    def createMetrics(label: String, data: RDD[LabeledPoint], model: ClassificationModel) = {
      val scoreAndLabels = data.map{
        point =>
          (model.predict(point.features), point.label)
      }
      val total = scoreAndLabels.map{ case (x, y) =>
        if (x == y) 1.0 else 0.0
      }.sum()
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
      (label, total / data.count(), metrics.areaUnderPR(), metrics.areaUnderROC())
    }

    val numIterations = 10
    val stepsize = 1.0
    val regParam = 0.0

    Seq(1, 5, 10, 50).map{param =>
      val model = trainWithParams(scaledData, regParam, param, new SimpleUpdater, stepsize)
      createMetrics(s"$param iterations", scaledData, model)
    }.foreach{
      case (str, acc, pr, roc) =>
        println(f"$str, ACC = ${acc * 100}%2.4f%%, PR = ${pr * 100}%2.4f%%, ROC = ${roc * 100}%2.4f%%")
    }

    Seq(0.001, 0.01, 0.1, 1.0, 10.0).map{param =>
      val model = trainWithParams(scaledData, regParam, numIterations, new SimpleUpdater, param)
      createMetrics(s"$param step size", scaledData, model)
    }.foreach{
      case (str, acc, pr, roc) =>
        println(f"$str, ACC = ${acc * 100}%2.4f%%, PR = ${pr * 100}%2.4f%%, ROC = ${roc * 100}%2.4f%%")
    }

    Seq(0.001, 0.01, 0.1, 1.0, 10.0).map{ param =>
      val model = trainWithParams(scaledData, param, numIterations, new SquaredL2Updater, stepsize)
      createMetrics(s"$param L2 regularisation parameter", scaledData, model)
    }.foreach {
      case (str, acc, pr, roc) =>
        println(f"$str, ACC = ${acc * 100}%2.4f%%, PR = ${pr * 100}%2.4f%%, ROC = ${roc * 100}%2.4f%%")
    }

    //check book for decision tree and naive bayes
  }

  def E6_CrossValidation() = {
    val data = E0_LoadData(d => d)
    val vectors = data.map(_.features)
    val scaler = new StandardScaler(withMean = true, withStd = true).fit(vectors)
    val scaledData = data.map(x => LabeledPoint(x.label, scaler.transform(x.features)))

    val trainTestSplit = scaledData.randomSplit(Array(0.6, 0.4), 123)
    val train = trainTestSplit(0)
    val test = trainTestSplit(1)

    def trainWithParams(input: RDD[LabeledPoint], regParam: Double, numIteration: Int, updater: Updater, stepSize: Double) = {
      val lr = new LogisticRegressionWithSGD()
      lr.optimizer.setNumIterations(numIteration).setRegParam(regParam).setUpdater(updater).setStepSize(stepSize)
      lr.run(input)
    }

    def createMetrics(label: String, data: RDD[LabeledPoint], model: ClassificationModel) = {
      val scoreAndLabels = data.map{
        point =>
          (model.predict(point.features), point.label)
      }
      val total = scoreAndLabels.map{ case (x, y) =>
        if (x == y) 1.0 else 0.0
      }.sum()
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
      (label, total / data.count(), metrics.areaUnderPR(), metrics.areaUnderROC())
    }

    val numIterations = 10
    val stepSize = 1.0
    val regParam = 0.0

    Seq(0.0, 0.001, 0.0025, 0.005, 0.01).map{ param =>
      val model = trainWithParams(train, param, numIterations, new SquaredL2Updater, stepSize)
      createMetrics(s"$param reg", train, model)
    }.foreach {
      case (str, acc, pr, roc) =>
        println(f"$str, ACC = ${acc * 100}%2.4f%%, PR = ${pr * 100}%2.4f%%, ROC = ${roc * 100}%2.4f%%")
    }

    Seq(0.0, 0.001, 0.0025, 0.005, 0.01).map{ param =>
      val model = trainWithParams(train, param, numIterations, new SquaredL2Updater, stepSize)
      createMetrics(s"$param reg", test, model)
    }.foreach {
      case (str, acc, pr, roc) =>
        println(f"$str, ACC = ${acc * 100}%2.4f%%, PR = ${pr * 100}%2.4f%%, ROC = ${roc * 100}%2.4f%%")
    }

  }

  def main(args: Array[String]) = {
    //E1_FourMethods
    //E2_ROCandAUC
    //E3_Scaling
    //E4_CategoryData
    //E5_TuneParam
    E6_CrossValidation
  }

}
