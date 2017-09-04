package Generator

import java.io._

import Utils.Control._

/**
  * Created by LD on 2017-02-14.
  */
object InputGenerator {

  val Separator = " "

  val SequentialPrefix = "src/main/resources/sample/sequential/2d/"

  val AkkaPrefix = "src/main/resources/sample/akka/2d/"

  def main(args: Array[String]): Unit = {

    var fileCount = 1
    val buf = new StringBuilder
    val bufAkka = new StringBuilder
    var xValue = 1
    var yValue = 1
    var totalValue = 0

    try {
      using(io.Source.fromFile("src/main/resources/GeneratorInput")) { source => {
        for (line <- source.getLines) {
          if (line.isEmpty) {

            val pw = new PrintWriter(new File(s"$SequentialPrefix$fileCount" ))
            val pw2 = new PrintWriter(new File(s"${SequentialPrefix}complete-$fileCount" ))
            pw.write(totalValue + " " + buf.toString())
            pw2.write(totalValue + " " + buf.toString())
            val pwAkka = new PrintWriter(new File(s"$AkkaPrefix$fileCount" ))
            val pwAkka2 = new PrintWriter(new File(s"${AkkaPrefix}complete-$fileCount" ))
            pwAkka.write(bufAkka.toString())
            pwAkka2.write(bufAkka.toString())
            pw.close()
            pw2.close()
            pwAkka.close()
            pwAkka2.close()

            buf.clear()
            bufAkka.clear()
            yValue = 1
            totalValue = 0
            fileCount += 1
          } else {
            line.toCharArray.foreach(c => {
              if (c != ' ') {
                buf.append(yValue)
                buf.append(Separator)
                buf.append(xValue)
                buf.append(Separator)
                buf.append(c)
                buf.append(Separator)
                totalValue += 1

                //akka style
                bufAkka.append(s"$yValue $xValue=$c\n")
              }
              xValue += 1
            })
            xValue = 1
            yValue += 1
          }
        }
      }}
    }
    catch {
      case e: Exception => println(e.getMessage)
    }
    println(s"Generated ${fileCount - 1} files!")
  }
}
