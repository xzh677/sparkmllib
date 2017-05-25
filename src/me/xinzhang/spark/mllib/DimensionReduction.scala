package me.xinzhang.spark.mllib

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

/**
  * Created by xin on 24/05/2017.
  */
object DimensionReduction {

  Logger.getLogger("org").setLevel(Level.ERROR)
  val sc = new SparkContext("local[*]", "DimensionReduction")



  def main(args: Array[String]) = {

    val path = "data/lfw"
    val rdd = sc.wholeTextFiles(path)
    val first = rdd.first
    println(first)

  }

}
