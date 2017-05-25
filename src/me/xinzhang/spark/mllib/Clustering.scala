package me.xinzhang.spark.mllib

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

/**
  * Created by xin on 24/05/2017.
  */
object Clustering {

  Logger.getLogger("org").setLevel(Level.ERROR)
  val sc = new SparkContext("local[*]", "Clustering")


  import org.apache.spark.mllib.linalg.Vectors

  def E0_FeatureExtraction() = {
    val movies = sc.textFile("data/ml-100k/u.item")
    val genres = sc.textFile("data/ml-100k/u.genre")
    val genreMap = genres.filter(!_.isEmpty).map(_.split("\\|")).map(array => (array(1), array(0))).collectAsMap()

    val titlesAndGenres = movies.map(_.split("\\|")).map { array =>
      val genres = array.toSeq.slice(5, array.size)
      val genresAssigned = genres.zipWithIndex.filter{
          case (g, idx) => g == "1"
      }.map { case (g, idx) =>
        genreMap(idx.toString)
      }
      (array(0).toInt, (array(1), genresAssigned))
    }
    //println(titlesAndGenres.first)

    import org.apache.spark.mllib.recommendation.ALS
    import org.apache.spark.mllib.recommendation.Rating
    val rawData = sc.textFile("data/ml-100k/u.data")
    val rawRatings = rawData.map(_.split("\t").take(3))
    val ratings = rawRatings.map{ case Array(user, movie, rating) =>
      Rating(user.toInt, movie.toInt, rating.toDouble)
    }
    ratings.cache()
    val alsModel = ALS.train(ratings, 50, 10, 0.1)


    val movieFactors = alsModel.productFeatures.map { case (id, factor) =>
      (id, Vectors.dense(factor))
    }

    val userFactors = alsModel.userFeatures.map { case (id, factor) =>
      (id, Vectors.dense(factor))
    }

    /*
    optional normalisation step.
    import org.apaache.spark.mllib.linalg.distributed.RowMatrix
    val matrix = new RowMatrix(userVectors)
    val matrixSummary = matrix.computeColumnSummaryStatistics()
     */
    (movieFactors, userFactors, titlesAndGenres)
  }


  import org.apache.spark.mllib.clustering.KMeans
  def E1_Clustering() = {
    val (movieFactors, userFactors, _) = E0_FeatureExtraction()
    val movieVectors = movieFactors.map(_._2)
    val userVectors = userFactors.map(_._2)

    val k = 5
    val numIterations = 100

    val movieClusterModel = KMeans.train(movieVectors, k, numIterations)

    val userClusterModel = KMeans.train(movieVectors, k, numIterations)

    val movie1 = movieVectors.first()
    println(movieClusterModel.predict(movie1))

    val predictions = movieClusterModel.predict(movieVectors)
    println(predictions.take(10).mkString("\t"))
  }

  import breeze.linalg._
  import breeze.numerics.pow

  def E2_Explaination() = {

    def computeDistance(v1: DenseVector[Double], v2: DenseVector[Double]) = pow(v1 - v2, 2).sum

    val (movieFactors, userFactors, titlesAndGenres) = E0_FeatureExtraction()
    val movieVectors = movieFactors.map(_._2)
    val userVectors = userFactors.map(_._2)

    val k = 5
    val numIterations = 100
    val movieClusterModel = KMeans.train(movieVectors, k, numIterations)

    //convert data to (id, ((title, genres), vector)) pairs
    val titleWithFactors = titlesAndGenres.join(movieFactors)

    //predict movie cluster and find cluster center, then find the Euclidean distance between them
    val moviesAssigned = titleWithFactors.map {
      case (id, ((title, genres), vector)) =>
        val pred = movieClusterModel.predict(vector)
        val clusterCenter = movieClusterModel.clusterCenters(pred)
        val dist = computeDistance(DenseVector(vector.toArray), DenseVector(clusterCenter.toArray))
        (id, title, genres.mkString(" "), pred, dist)
    }

    //convert to a map with (k -> (id, title, genres, cluster, dist))
    val clusterAssignments = moviesAssigned.groupBy{
      case (id, title, genres, cluster, dist) => cluster
    }.collectAsMap()

    //iteration through each cluster from k=0 to k=4 in _._1
    for ((k, v) <- clusterAssignments.toSeq.sortBy(_._1)) {
      println(f"Cluster $k:")
      val m = v.toSeq.sortBy(_._5)
      println(m.take(20).map {case (_, title, genres, _, d) =>
        (title, genres, d)
      }.mkString("\n"))
      println("============\n")
    }
  }

  def E3_Tuning() = {
    val (movieFactors, userFactors, _) = E0_FeatureExtraction()
    val movieVectors = movieFactors.map(_._2)
    val userVectors = userFactors.map(_._2)

    val k = 5
    val numIterations = 100

    val movieClusterModel = KMeans.train(movieVectors, k, numIterations)

    val userClusterModel = KMeans.train(movieVectors, k, numIterations)

    println("Within cluster sum of squre")
    println("for movies: " + movieClusterModel.computeCost(movieVectors))
    println("for users: " + userClusterModel.computeCost(userVectors))

    val trainTestSplitMovies = movieVectors.randomSplit(Array(0.6, 0.4), 123)
    val trainMovies = trainTestSplitMovies(0)
    val testMovies = trainTestSplitMovies(1)
    println("Movie clustering cross-validation")
    Seq(2, 3, 4, 5, 10, 20).map { k =>
      (k, KMeans.train(trainMovies, k, numIterations).computeCost(testMovies))
    }.foreach{
      case (k, cost) => println(f"WCCS for K=$k is $cost%2.2f")
    }
  }

  def main(args: Array[String]) = {
    //E1_Clustering
    //E2_Explaination
    E3_Tuning
  }
}
