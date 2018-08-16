package mllib

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils

object SVMWithSGDExample {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("SVMWithSGDExample").setMaster("local[2]")
    val sc = new SparkContext(conf)

    // 加载和解析数据文件
    val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")

    // 将数据且分成训练集和测试集 (40%用来测试)
    val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    // numIterations表示方法单次运行需要迭代的次数
    // 生成模型
    val numIterations = 100
    val model = SVMWithSGD.train(training, numIterations)

    // 清空默认的阈值
    model.clearThreshold()

    // 用测试集评估模型
    val scoreAndLabels = test.map { point =>
      val score = model.predict(point.features)
      (score, point.label)
    }

    // 获取评估矩阵
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    // AUC值是一个概率值，当你随机挑选一个正样本以及负样本，
    // 当前的分类算法根据计算得到的Score值将这个正样本排在负样本前面的概率就是AUC值，AUC值越大，
    // 当前分类算法越有可能将正样本排在负样本前面，从而能够更好地分类
    val auROC = metrics.areaUnderROC()

    println("Area under ROC = " + auROC)

    // 保存模型，及后续使用模型时只需加载模型即可，无需再次训练
    model.save(sc, "target/tmp/scalaSVMWithSGDModel")
    val sameModel = SVMModel.load(sc, "target/tmp/scalaSVMWithSGDModel")

    sc.stop()
  }
}
