package com.github.ivetan.spark

import org.apache.spark.ml.regression.LinearRegression

object ExerciseJul17 extends App {
  //TODO load range3d.csv

  //TODO Find Interecept and Coefficients (there are 3!) for simple Linear Regression

  val spark = SparkUtil.createSpark("range3d")
  val filePath = "./src/resources/csv/range3d.csv"
  val df = spark.read
    .format("csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .load(filePath)

  df.printSchema()
  df.describe().show(false)
  df.show(5)

  import org.apache.spark.ml.feature.RFormula
  val supervised = new RFormula()
    .setFormula("y ~ .")

  val ndf = supervised
    .fit(df)
    .transform(df)

  ndf.show(5)

  val linReg = new LinearRegression()

  val lrModel = linReg.fit(ndf)

  val intercept = lrModel.intercept
  val coefficients = lrModel.coefficients
  val x1 = coefficients(0)
  val x2 = coefficients(1)
  val x3 = coefficients(2)

  println(s"Intercept: $intercept, coefficients: x1 = $x1,  x2 = $x2 and x3 = $x3")

}
