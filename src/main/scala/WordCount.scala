/**
  * Created by gxy on 18-4-16.
  */
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.log4j.{Level,Logger}

object WordCount {
  def main(args: Array[String]) {
    //屏蔽日志
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
    //（你的输入文件路径）
    val inputFile =  "/home/gxy/input/wc.txt"
    val conf = new SparkConf().setAppName("WordCount")
      //setMaster("local") 本机的spark就用local，远端的就写ip
      //如果是打成jar包运行则需要去掉 setMaster("local")因为在参数中会指定。
      .setMaster("local[2]")
    val sc = new SparkContext(conf)
    val textFile = sc.textFile(inputFile)
    val wordCount = textFile.flatMap(line => line.split(" ")).map(word => (word, 1)).reduceByKey((a, b) => a + b)
    wordCount.foreach(println)
  }
}
