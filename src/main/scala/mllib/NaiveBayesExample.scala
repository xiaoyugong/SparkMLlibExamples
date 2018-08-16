package mllib

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.util.MLUtils


object NaiveBayesExample {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("NaiveBayesExample").setMaster("local[2]")
    val sc = new SparkContext(conf)

    // 加载和解析数据文件
    // 每个RDD元素代表一个数据点，每个数据点包含，标签和数据特征，对分类来讲标签的值是{0, 1, ..., numClasses-1}
    val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")

    // 将数据且分成训练集和测试集 (40%用来测试)
    val Array(training, test) = data.randomSplit(Array(0.6, 0.4))

    // lambda是一个加法平滑参数，默认值是1.0
    // modelType指定是使用multinomial还是bernoulli算法模型，默认是multinomial
    // 生成模型
    val model = NaiveBayes.train(training, lambda = 1.0, modelType = "multinomial")

    // 用测试集评估模型
    val predictionAndLabel = test.map(p => (model.predict(p.features), p.label))
    // 计算精度
    val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()
    println("Accuracy = " + accuracy)

    // 保存模型，及后续使用模型时只需加载模型即可，无需再次训练
    model.save(sc, "target/tmp/myNaiveBayesModel")
    val sameModel = NaiveBayesModel.load(sc, "target/tmp/myNaiveBayesModel")
  }
}

