package mllib
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.clustering.{DistributedLDAModel, LDA}
import org.apache.spark.mllib.linalg.Vectors
/**
  * Created by gxy on 18-5-3.
  */
object LDA {
  def main(args: Array[String]) {
    // 设置运行环境
    val conf = new SparkConf().setAppName("Kmeans").setMaster("local[2]")
    val sc = new SparkContext(conf)

    // Load and parse the data
    val data = sc.textFile("data/mllib/sample_lda_data.txt")
    val parsedData = data.map(s => Vectors.dense(s.trim.split(' ').map(_.toDouble)))
    // Index documents with unique IDs
    val corpus = parsedData.zipWithIndex.map(_.swap).cache()

    // Cluster the documents into three topics using LDA
    val ldaModel = new LDA().setK(3).run(corpus)

    // Output topics. Each is a distribution over words (matching word count vectors)
    println(s"Learned topics (as distributions over vocab of ${ldaModel.vocabSize} words):")
    val topics = ldaModel.topicsMatrix
    for (topic <- Range(0, 3)) {
      print(s"Topic $topic :")
      for (word <- Range(0, ldaModel.vocabSize)) {
        print(s"${topics(word, topic)}")
      }
      println()
    }

    // Save and load model.
    ldaModel.save(sc, "target/org/apache/spark/LatentDirichletAllocationExample/LDAModel")
    val sameModel = DistributedLDAModel.load(sc,
      "target/org/apache/spark/LatentDirichletAllocationExample/LDAModel")
  }
}
