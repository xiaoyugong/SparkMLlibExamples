package mllib

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LinearRegressionModel
import org.apache.spark.mllib.regression.LinearRegressionWithSGD

object LinearRegressionWithSGDExample {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("LinearRegressionWithSGDExample").setMaster("local[2]")
    val sc = new SparkContext(conf)

    // 加载和解析数据文件
    val data = sc.textFile("data/mllib/ridge-data/lpsa.data")
    val parsedData = data.map { line =>
      val parts = line.split(',')
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
    }.cache()

    // stepSize表示每次迭代的步长
    // numIterations表示方法单次运行需要迭代的次数
    // 生成模型
    val numIterations = 100
    val stepSize = 0.00000001
    val model = LinearRegressionWithSGD.train(parsedData, numIterations, stepSize)

    // 评估模型
    val valuesAndPreds = parsedData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    //计算均平方误差
    val MSE = valuesAndPreds.map{ case(v, p) => math.pow(v - p, 2) }.mean()
    println("training Mean Squared Error = " + MSE)

    // 保存模型，及后续使用模型时只需加载模型即可，无需再次训练
    model.save(sc, "target/tmp/scalaLinearRegressionWithSGDModel")
    val sameModel = LinearRegressionModel.load(sc, "target/tmp/scalaLinearRegressionWithSGDModel")

    sc.stop()
  }
}
