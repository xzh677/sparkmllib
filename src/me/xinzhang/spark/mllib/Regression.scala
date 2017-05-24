package me.xinzhang.spark.mllib

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

/**
  * Created by xin on 24/05/2017.
  */
object Regression {

  Logger.getLogger("org").setLevel(Level.ERROR)
  val sc = new SparkContext("local[*]", "Regression")

  def E1_Feature() = {
    val rawData = sc.textFile("data/Bike-Sharing-Dataset/hour_nohead.csv")
    val records = rawData.map(x => x.split(",")).cache()

    def getMapping(records: RDD[Array[String]], idx: Int) = {
      records.map(fields => fields(idx)).distinct.zipWithIndex.collectAsMap()
    }
    println(s"Mapping of first categorical feature column: ${getMapping(records, 2)}")
  }

  def main(args:Array[String]) = {
    E1_Feature()
  }
}
