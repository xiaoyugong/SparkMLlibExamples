import org.apache.spark.mllib.feature.Normalizer
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.clustering.{BisectingKMeans, KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.sql.SparkSession

/**
  * Created by gxy on 18-5-3.
  */
object BehaivorCluster {
    def main(args: Array[String]) {
      // 设置运行环境
      val sparkSession: SparkSession = SparkSession.builder().appName("Kmeans")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .master("local[2]").getOrCreate()
      val sc = sparkSession.sparkContext

      // 加载和解析数据文件
//      val data02 = sc.textFile("/home/gxy/clusterData/201802/part-00000")
//      val data03 = sc.textFile("/home/gxy/clusterData/201803/part-00000")
//      val data04 = sc.textFile("/home/gxy/clusterData/201804/part-00000")
//      val data = data02.union(data03).union(data04)
      val data = sc.textFile("/home/gxy/weekTrainCluster/week_cluster_data_of_m_final")
      val data2 = sc.textFile("/home/gxy/weekTrainCluster/week_cluster_test_data")
//      val data = sc.textFile("/home/gxy/clusterData/2018234/part-00000")
      val parsedData = data.map(s => Vectors.dense(s.split(",")(1).replace(")","").split("\\|").map(_.toDouble)))

      // 正则化每个向量到1阶范数
      val normalizer = new Normalizer()

      val l1NormData = normalizer.transform(parsedData)
      l1NormData.foreach(println)
      // 将数据集聚类，5个类，20次迭代，进行模型训练形成数据模型
      val numClusters = 5
      val numIterations = 500
      val clusters = KMeans.train(l1NormData, numClusters, numIterations)

      // 打印数据模型的中心点
      println("Cluster centers:")
      for (c <- clusters.clusterCenters) {
        println(c.toString)
      }

      // 使用误差平方之和来评估数据模型
      val WSSSE = clusters.computeCost(l1NormData)
      println("Within Set Sum of Squared Errors = " + WSSSE)

      // 交叉评估，返回数据集和结果
      data.map(line => {
        val lineVector = Vectors.dense(line.split(",")(1).replace(")","").split("\\|").map(_.toDouble))
        val nor = normalizer.transform(lineVector)
        val prediction = clusters.predict(nor)
        line.split(",")(0).replace("(","") + "," + line.split(",")(1).replace(")","") + "," + prediction
      }).repartition(1)
//        .map(r=>(r.split(",")(0).split("_")(0), r.split(",")(1)))
//        .reduceByKey((x, y) => x)
        .saveAsTextFile("target/tmp/weekBehaivorWOfM_final")

      data2.map(line => {
        val lineVector = Vectors.dense(line.split(",")(1).replace(")","").split("\\|").map(_.toDouble))
        val nor = normalizer.transform(lineVector)
        val prediction = clusters.predict(nor)
        line.replace("(","").replace(")","") + "," + prediction
      }).repartition(1).saveAsTextFile("target/tmp/weekBehaivorWOfM_final/test")
//
//      data03.map(line => {
//        val lineVector = Vectors.dense(line.split(",")(1).replace(")","").split("\\|").map(_.toDouble))
//        val nor = normalizer.transform(lineVector)
//        val prediction = clusters.predict(nor)
//        line.replace("(","").replace(")","") + "," + prediction
//      }).repartition(1).saveAsTextFile("target/tmp/behaivor2/03")
//
//      data04.map(line => {
//        val lineVector = Vectors.dense(line.split(",")(1).replace(")","").split("\\|").map(_.toDouble))
//        val nor = normalizer.transform(lineVector)
//        val prediction = clusters.predict(nor)
//        line.replace("(","").replace(")","") + "," + prediction
//      }).repartition(1).saveAsTextFile("target/tmp/behaivor2/04")
      // 保存模型，及后续使用模型时只需加载模型即可，无需再次训练
      clusters.save(sc, "target/tmp/KMeansExample/KMeansModel")
//      val sameModel = KMeansModel.load(sc, "target/tmp/KMeansExample/KMeansModel")
//      data2.map(line => {
//        val lineVector = Vectors.dense(line.split(",")(1).replace(")","").split("\\|").map(_.toDouble))
//        val nor = normalizer.transform(lineVector)
//        val prediction = sameModel.predict(nor)
//        line.replace("(","").replace(")","") + "," + prediction
//      }).repartition(1).saveAsTextFile("target/tmp/weekBehaivorWOfM_final/test")
      sc.stop()
    }
}
