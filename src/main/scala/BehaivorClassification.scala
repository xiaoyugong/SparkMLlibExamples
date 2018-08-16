import org.apache.spark.sql.{SparkSession, functions}

/**
  * Created by gxy on 18-5-22.
  */
object BehaivorClassification {
  def main(args: Array[String]): Unit = {
    // 设置运行环境
    val sparkSession: SparkSession = SparkSession.builder().appName("Kmeans")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .master("local[2]").getOrCreate()
    val sc = sparkSession.sparkContext

    val sqlContext = sparkSession.sqlContext
    import sqlContext.implicits._

    // 加载和解析数据文件
    val data = sc.textFile("/home/gxy/weekTrainCluster/week_cluster_data_of_m_copy")
    val dataDF = data.map(r => {
      val split = r.split(",")
      val valueList = split(1).split("\\|")
      (split(0).replace("(",""), valueList(0), valueList(1), valueList(2), valueList(3), valueList(4), valueList(5), valueList(6), valueList(7), valueList(8).replace(")",""))
    }).toDF("ip", "news", "job", "funny", "life", "otherIM", "professional", "game", "QQ", "weiXin")

    dataDF.show()

    val meanDF = dataDF.agg(
      functions.mean("news").alias("news"),
      functions.mean("job").alias("job"),
      functions.mean("funny").alias("funny"),
      functions.mean("life").alias("life"),
      functions.mean("otherIM").alias("otherIM"),
      functions.mean("professional").alias("professional"),
      functions.mean("game").alias("game"),
      functions.mean("QQ").alias("QQ"),
      functions.mean("weiXin").alias("weiXin")
    )

    val mean = meanDF.select($"news" + $"funny" + $"life" + $"game" as "nonWork", $"professional").collectAsList().get(0).toSeq
    println(mean)

    dataDF.rdd.map(r => {
      var label = ""
      val ip = r.get(0)
      val work = r.get(6).toString.toInt
      val nonWork = r.get(1).toString.toInt + r.get(3).toString.toInt + r.get(4).toString.toInt + r.get(7).toString.toInt
      val im = r.get(5).toString.toInt + r.get(8).toString.toInt + r.get(9).toString.toInt
      if (work > mean(1).asInstanceOf[Double]) label += "G" else label += "H"

      if (nonWork > mean.head.asInstanceOf[Double]) label += "E" else label += "F"

      if (nonWork > 2 * work) label += "A" else label += "B"

      if (im > 10000) label += "C" else label += "D"

      val finalLabel = label match {
        case "GEAC" => 0
        case "GEAD" => 1
        case "GEBC" => 2
        case "GEBD" => 3
        case "GFAC" => 4
        case "GFAD" => 5
        case "GFBC" => 6
        case "GFBD" => 7
        case "HEAC" => 8
        case "HEAD" => 9
        case "HEBC" => 10
        case "HEBD" => 11
        case "HFAC" => 12
        case "HFAD" => 13
        case "HFBC" => 14
        case "HFBD" => 15
      }
      ip + "," + finalLabel
    }).repartition(1).saveAsTextFile("target/tmp/ip_label16")
  }
}
