package me.xinzhang.spark.mllib

import java.nio.charset.CodingErrorAction

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.recommendation._
import scala.io.Source
import scala.io.Codec
import org.jblas.DoubleMatrix

/**
  * Created by xin on 22/05/2017.
  */
object Recommendations {

  private val sc = new SparkContext("local[*]", "Recommendations")

  Logger.getLogger("org").setLevel(Level.ERROR)

  def loadMovieNames(filename: String): Map[Int, String] = {
    implicit val codec = Codec("UTF-8")
    codec.onMalformedInput(CodingErrorAction.REPLACE)
    codec.onUnmappableCharacter(CodingErrorAction.REPLACE)
    var movieNames: Map[Int, String] = Map()
    val lines = Source.fromFile(filename).getLines()
    lines.foreach(line => {
      val fields = line.split('|')
      if (fields.length > 1) {
        movieNames += (fields(0).toInt -> fields(1))
      }
    })
    movieNames
  }

  def loadRatings(filename: String): RDD[Rating] = {
    val lines = sc.textFile(filename)
    lines.map(x => x.split('\t')).map(x => Rating(x(0).toInt, x(1).toInt, x(2).toDouble))
  }

  def trainModel(ratings:RDD[Rating]): MatrixFactorizationModel = {


    val rank = 50
    val iteration = 10
    val lambda = 0.01
    //use alternating least square to train the model (matrix factorization)
    val model = ALS.train(ratings, rank, iteration, lambda)
    println("User feature size: " + model.userFeatures.count() + " * " + model.userFeatures.take(1)(0)._2.length)
    println("Product feature size:" + model.productFeatures.take(1)(0)._2.length + " * " + model.productFeatures.count())
    model
  }

  def E1_recommendations(): Unit = {
    val nameDict = loadMovieNames("data/ml-100k/u.item")
    val ratings = loadRatings("data/ml-100k/u1.base").cache()
    val model = trainModel(ratings)

    val userID = 789
    val productID = 123
    val topK = 10
    def printMovieTitleRating(rating: Rating): Unit = {
      println(nameDict(rating.product) + ": " + rating.rating)
    }

    println("\nUser " + topK + "'s top " + topK + " rated movies:")
    val moviesForUser = ratings.keyBy(_.user).lookup(userID)
    //-_.rating means sort by rating in descending order
    moviesForUser.sortBy(-_.rating).take(10).foreach(printMovieTitleRating)

    println("\nUser " + userID + " will rate " + nameDict(productID) + ": " + model.predict(userID, productID))

    println("\nTop " + topK + " recommendations for user " + userID + ":")
    model.recommendProducts(userID, topK).foreach(printMovieTitleRating)
  }

  def E2_productRecommendations(): Unit = {
    val nameDict = loadMovieNames("data/ml-100k/u.item")
    val ratings = loadRatings("data/ml-100k/u1.base").cache()
    val model = trainModel(ratings)

    def cosineSimilarity(vec1:DoubleMatrix, vec2: DoubleMatrix): Double = {
      vec1.dot(vec2) / (vec1.norm2() * vec2.norm2())
    }
    val productID = 50
    val topK = 10

    println("Top " + topK + " movies similar to " + nameDict(productID) + ":")
    val productFeature = model.productFeatures.lookup(productID).head
    val productVector = new DoubleMatrix(productFeature)
    val sims = model.productFeatures.map{ case (id, feature) => {
      val featureVector = new DoubleMatrix(feature)
      val sim = cosineSimilarity(featureVector, productVector)
      (id, sim)
    }}.sortBy(-_._2).take(topK)
    sims.foreach(x => println(nameDict(x._1) + ": " + x._2))
  }

  def E3_modelEvaluation(): Unit = {
    val nameDict = loadMovieNames("data/ml-100k/u.item")
    val ratings = loadRatings("data/ml-100k/u1.base").cache()
    val rank = 50
    val numIterations = 10
    var lambdas = List(0.05,0.01,0.005,0.001,0.005)

    def computeError(ratings: RDD[Rating], model: MatrixFactorizationModel) : (Double, Double) = {
      val userProducts = ratings.map{
        case Rating(user, product, rating) => (user, product)
      }
      val predictions = model.predict(userProducts).map{
        case Rating(user, product, rating) => ((user, product), rating)
      }
      val ratingsAndPredictions = ratings.map{
        case Rating(user, product, rating) => ((user, product), rating)
      }.join(predictions)
      // ratingsAndPredictions is ((user, product), (actual, predict))

      import org.apache.spark.mllib.evaluation.RegressionMetrics
      val predictedAndTrue = ratingsAndPredictions.map{
        case ((user, product), (actual, predict)) => (predict, actual)
      }
      val regressionMetrics = new RegressionMetrics(predictedAndTrue)
      (regressionMetrics.meanSquaredError, regressionMetrics.rootMeanSquaredError)
    }

    val models = lambdas.map(x => {
      val model = ALS.train(ratings, rank, numIterations, x)
      (x, model, computeError(ratings, model))
    })
    models.foreach(println)
    println("best model:")
    println(models.minBy(_._3._1))
  }

  def main(args: Array[String]): Unit = {
    E3_modelEvaluation()
  }

}
