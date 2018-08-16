package mllib

import org.apache.spark.{SparkConf, SparkContext}
// $example on$
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils

object LogisticRegressionWithLBFGSExample {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("LogisticRegressionWithLBFGSExample").setMaster("local[2]")
    val sc = new SparkContext(conf)

    // Load training data in LIBSVM format.
    val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")

    // Split data into training (60%) and test (40%).
    val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    // Run training algorithm to build the model
    val model = new LogisticRegressionWithLBFGS()
      .setNumClasses(10)
      .run(training)

    // Compute raw scores on the test set.
    val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }

    // Get evaluation metrics.
    val metrics = new MulticlassMetrics(predictionAndLabels)
    val accuracy = metrics.accuracy
    println(s"metrics = $metrics")
    println(s"Accuracy = $accuracy")

    // Save and load model
    model.save(sc, "target/tmp/scalaLogisticRegressionWithLBFGSModel")
    val sameModel = LogisticRegressionModel.load(sc,
      "target/tmp/scalaLogisticRegressionWithLBFGSModel")

    sc.stop()
  }
}
