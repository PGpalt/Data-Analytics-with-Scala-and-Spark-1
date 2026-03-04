import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object Main extends App
{

  val spark = SparkSession.builder.master("local[*]")
    .appName("ScalaSparkProject2015030049")
    .getOrCreate()
  val sc = spark.sparkContext

  val hdfsURI = "hdfs://localhost:9000"
  FileSystem.setDefaultUri(spark.sparkContext.hadoopConfiguration, hdfsURI)
  val hdfs = FileSystem.get(spark.sparkContext.hadoopConfiguration)

  val outputPath = "/output"
  val categoryFilePath = "hdfs://localhost:9000/input/rcv1-v2.topics.qrels"
  val termFilePath1 = "/input/lyrl2004_vectors_test_pt0.dat"
  val termFilePath2 = "/input/lyrl2004_vectors_test_pt1.dat"
  val termFilePath3 = "/input/lyrl2004_vectors_test_pt2.dat"
  val termFilePath4 = "/input/lyrl2004_vectors_test_pt3.dat"
  val termFilePath5 = "/input/lyrl2004_vectors_train.dat"
  val stemFilePath = "/input/stem.termid.idf.map.txt"

//  val spark = SparkSession.builder
//    .appName("ScalaSparkProject2015030049")
//      .master("yarn")
//      .config("spark.hadoop.fs.defaultFS", "hdfs://clu01.softnet.tuc.gr:8020")
//      .config("spark.hadoop.yarn.resourcemanager.address", "http://clu01.softnet.tuc.gr:8189")
//      .config("spark.hadoop.yarn.application.classpath", "$HADOOP_CONF_DIR,$HADOOP_COMMON_HOME/*,$HADOOP_COMMON_HOME/lib/*,$HADOOP_HDFS_HOME/*,$HADOOP_HDFS_HOME/lib/*,$HADOOP_MAPRED_HOME/*,$HADOOP_MAPRED_HOME/lib/*,$HADOOP_YARN_HOME/*,$HADOOP_YARN_HOME/lib/*")
//    .getOrCreate()
//
//  val hdfsURI = "hdfs://clu01.softnet.tuc.gr:8020"
//  FileSystem.setDefaultUri(spark.sparkContext.hadoopConfiguration, hdfsURI)
//  val hdfs = FileSystem.get(spark.sparkContext.hadoopConfiguration)
//
//  val sc = spark.sparkContext
//
//
//  val outputPath = "/user/fp21/out"
//  val categoryFilePath = "/user/chrisa/Reuters/rcv1-v2.topics.qrels"
//  val termFilePath1 = "/user/chrisa/Reuters/lyrl2004_vectors_test_pt0.dat"
//  val termFilePath2 = "/user/chrisa/Reuters/lyrl2004_vectors_test_pt1.dat"
//  val termFilePath3 = "/user/chrisa/Reuters/lyrl2004_vectors_test_pt2.dat"
//  val termFilePath4 = "/user/chrisa/Reuters/lyrl2004_vectors_test_pt3.dat"
//  val termFilePath5 = "/user/chrisa/Reuters/lyrl2004_vectors_train.dat"
//  val stemFilePath = "/user/chrisa/Reuters/stem.termid.idf.map.txt"

  val termFileRDD = sc.textFile(termFilePath1)
    .union(sc.textFile(termFilePath2))
    .union(sc.textFile(termFilePath3))
    .union(sc.textFile(termFilePath4))
    .union(sc.textFile(termFilePath5))

  val DOC_CAT_COUNT = sc.textFile(categoryFilePath).map(line => (line.split("\\s+")
      .headOption
      .getOrElse().toString , 1))
    .reduceByKey(_ + _)

  val DOC_TERM_COUNT = termFileRDD.flatMap(_.split(":"))
    .flatMap(_.split("\\s+").tail)
    .map(term => (term , 1) )
    .reduceByKey(_ + _)

  val DOC_C = sc.textFile(categoryFilePath).map(line => (line.split("\\s+").tail.headOption.getOrElse().toString ,
    line.split("\\s+")
      .headOption
      .getOrElse().toString))

  val DOC_T = termFileRDD.flatMap { line =>
    val words = line.split(" ")
    val docID = words(0)
    val DocTermPair = words.drop(1)

    DocTermPair.map { t =>
      val termID = t.split(":")(0)
      (docID, termID)
    }.filterNot(pair => pair._2.isEmpty)
  }

  val DOCintersectionCount = DOC_C.join(DOC_T)
    .map( x => (x._2,1) )
    .reduceByKey(_ + _)

  val jaccardIndex = DOCintersectionCount.map( x => (x._1._1,(x._1._2,x._2)))
    .join(DOC_CAT_COUNT)
    .map( x => (x._2._1._1,(x._1,x._2._1._2,x._2._2)))
    .join(DOC_TERM_COUNT)
    .map( x => (x._1,(x._2._1._1,x._2._1._2.toDouble / (x._2._1._3 + x._2._2 - x._2._1._2)))) // calculating Jaccard Index

  val finalResult = sc.textFile(stemFilePath).map(line =>
      (line.split("\\s+")(1) , line.split("\\s+")(0)))
    .join(jaccardIndex)
    .map( x => (x._2._2._1,x._2._1,x._2._2._2))

  hdfs.delete(new Path(outputPath), true)
  finalResult.saveAsTextFile(outputPath)

  spark.stop()
}