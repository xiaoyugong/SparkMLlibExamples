package mllib

import org.apache.spark.mllib.fpm.{FPGrowth, FPGrowthModel}
import org.apache.spark.{SparkConf, SparkContext}

object FPGrowthExample {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("FPGrowthTest").setMaster("local[2]")
    val sc = new SparkContext(conf)

    //取出数据
    val data = sc.textFile("data/mllib/sample_fpgrowth.txt")

    //把数据通过空格分割
    val transactions = data.map(x => x.split(" "))
    transactions.cache()

    //设置参数
    val minSupport = 0.2 //最小支持度
    val minConfidence = 0.8 //最小置信度
    val numPartitions = 2 //数据分区数
    //创建一个FPGrowth的算法实列
    val fpg = new FPGrowth()
    fpg.setMinSupport(minSupport)
    fpg.setNumPartitions(numPartitions)

    //使用样本数据建立模型
    val model = fpg.run(transactions)

    //查看所有的频繁项集，并且列出它出现的次数
    model.freqItemsets.collect().foreach(itemset => {
      println(itemset.items.mkString("[", ",", "]") + "," + itemset.freq)
    })

    //通过置信度筛选出推荐规则
    //antecedent表示前项，consequent表示后项
    //confidence表示规则的置信度
    model.generateAssociationRules(minConfidence).collect().foreach(rule => {
      println(rule.antecedent.mkString(",") + "-->" +
        rule.consequent.mkString(",") + "-->" + rule.confidence)
    })

    //查看规则生成的数量
    println(model.generateAssociationRules(minConfidence).collect().length)

    // 保存模型，及后续使用模型时只需加载模型即可，无需再次训练
    model.save(sc, "target/tmp/ALSModel")
    val sameModel = FPGrowthModel.load(sc, "target/tmp/ALSModel")

    sc.stop()
  }
}