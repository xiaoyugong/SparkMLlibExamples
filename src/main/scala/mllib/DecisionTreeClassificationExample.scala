package mllib

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.util.MLUtils

object DecisionTreeClassificationExample {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("DecisionTreeClassificationExample").setMaster("local[2]")
    val sc = new SparkContext(conf)

    // 加载和解析数据文件
    // 每个RDD元素代表一个数据点，每个数据点包含，标签和数据特征，对分类来讲标签的值是{0, 1, ..., numClasses-1}
    val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
    // 将数据且分成训练集和测试集 (30%用来测试)
    val splits = data.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    // 训练决策树模型
    // 空的categoricalFeaturesInfo（分类特征信息）表明所有特征都是连续的
    // numClasses：表示分类的数量，默认值是2
    // categoricalFeaturesInfo：存储离散性属性的映射关系，比如（5—>4）表示数据点的第5个特征是离散性属性，有4个类别，取值为{0，1，2，3}
    // impurity：表示信息纯度的计算方法，包括Gini参数或信息熵
    // maxDepth：表示树的最大深度
    // maxBins：表示分类属性的最大值
    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val impurity = "gini"
    val maxDepth = 5
    val maxBins = 32

    // 生成的决策树模型
    val model = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
      impurity, maxDepth, maxBins)

    // 用测试集评估模型
    val labelAndPreds = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val testErr = labelAndPreds.filter(r => r._1 != r._2).count().toDouble / testData.count()
    println("Test Error = " + testErr)
    println("Learned classification tree model:\n" + model.toDebugString)

    // 保存模型，及后续使用模型时只需加载模型即可，无需再次训练
    model.save(sc, "target/tmp/myDecisionTreeClassificationModel")
    val sameModel = DecisionTreeModel.load(sc, "target/tmp/myDecisionTreeClassificationModel")
  }
}
