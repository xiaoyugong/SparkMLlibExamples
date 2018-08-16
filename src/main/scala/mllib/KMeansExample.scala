package mllib

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors

object KMeansExample {
  def main(args: Array[String]) {
    // 设置运行环境
    val conf = new SparkConf().setAppName("Kmeans").setMaster("local[2]")
    val sc = new SparkContext(conf)

    // 加载和解析数据文件
    val data = sc.textFile("data/mllib/kmeans_data.txt")
    val parsedData = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble)))

    // 将数据集聚类，2个类，20次迭代，进行模型训练形成数据模型
    val numClusters = 2
    val numIterations = 20
    val clusters = KMeans.train(parsedData, numClusters, numIterations)

    // 打印数据模型的中心点
    println("Cluster centers:")
    for (c <- clusters.clusterCenters) {
      println("" + c.toString)
    }

    // 使用误差平方之和来评估数据模型
    val WSSSE = clusters.computeCost(parsedData)
    println("Within Set Sum of Squared Errors = " + WSSSE)

    // 使用模型测试单点数据
    println("Vectors 0.2 0.2 0.2 is belongs to clusters:" +
      clusters.predict(Vectors.dense("0.2 0.2 0.2".split(' ').map(_.toDouble))))
    println("Vectors 0.25 0.25 0.25 is belongs to clusters:" +
      clusters.predict(Vectors.dense("0.25 0.25 0.25".split(' ').map(_.toDouble))))
    println("Vectors 8 8 8 is belongs to clusters:" +
      clusters.predict(Vectors.dense("8 8 8".split(' ').map(_.toDouble))))
    // 交叉评估1，只返回结果
    val testdata = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble)))
    val result1 = clusters.predict(testdata)
    result1.saveAsTextFile("target/tmp/KMeansExample/result_kmeans1")
    // 交叉评估2，返回数据集和结果
    val result2 = data.map(line => {
      val linevectore = Vectors.dense(line.split(' ').map(_.toDouble))
      val prediction = clusters.predict(linevectore)
      line + " " + prediction
    }).saveAsTextFile("target/tmp/KMeansExample/result_kmeans2")

    // 保存模型，及后续使用模型时只需加载模型即可，无需再次训练
    clusters.save(sc, "target/tmp/KMeansExample/KMeansModel")
    val sameModel = KMeansModel.load(sc, "target/tmp/KMeansExample/KMeansModel")
    sc.stop()
  }
}