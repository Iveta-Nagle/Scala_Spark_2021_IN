package com.github.ivetan.spark


import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.RFormula
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.sql.DataFrame

object ExerciseJul15 extends App {

  val spark = SparkUtil.createSpark("irisesClassification")
  val filePath = "./src/resources/irises/iris.data"
  val df = spark.read
    .format("csv")
    .option("inferSchema", "true")
    .load(filePath)

  df.printSchema()
  df.describe().show(false)
  df.show(5)

  val supervised = new RFormula()
    .setFormula("flower ~ . ")

  val ndf = df.withColumnRenamed("_c4", "flower")
  ndf.show(5, false)

  val fittedRF = supervised.fit(ndf)
  val preparedDF = fittedRF.transform(ndf)
  preparedDF.show(false)
  preparedDF.sample(0.1).show(false)

  val Array(train, test) = preparedDF.randomSplit(Array(0.75, 0.25))

  val randForest = new RandomForestRegressor()
    .setLabelCol("label")
    .setFeaturesCol("features")

  val fittedModel = randForest.fit(train)

  val testDF = fittedModel.transform(test)

  testDF.show(30, false)

  def showAccuracy(df: DataFrame): Unit = {
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(df)
    println(s"DF size: ${df.count()} Accuracy $accuracy - Test Error = ${1.0 - accuracy}")
  }

  showAccuracy(testDF)

}
