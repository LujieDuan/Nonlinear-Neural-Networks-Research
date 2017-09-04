package Utils

import breeze.linalg.{DenseMatrix, DenseVector}

import scala.io.StdIn

/**
  * Created by LD on 2017-03-01.
  */
class NetworkBuilder(firstLayerSize: Int, lastLayerSize: Int) {

  var Layers: Seq[Int] = _
  var LambdaOfC: Seq[Seq[Double]] = _
  var LambdaOfE: Seq[Seq[Double]] = _

  def buildNetwork: Unit = {
    println("How many layers?")
    val numberOfLayers = StdIn.readInt()
    val layers = new Array[Int](numberOfLayers)
    val lambdaOfC = new Array[Array[Double]](numberOfLayers)
    val lambdaOfE = new Array[Array[Double]](numberOfLayers)

    for (i <- 0 until numberOfLayers) {
      if(i == 0) layers(i) = firstLayerSize
      else if(i == numberOfLayers - 1) layers(i) = lastLayerSize
      else {
        println(s"How many neurons at layer $i?(Not including input layer)")
        layers(i) = StdIn.readInt()
      }
      val layerLambdaOfC = new Array[Double](layers(i))
      val layerLambdaOfE = new Array[Double](layers(i))
      for (j <- 0 until layers(i)) {
/*        if (i != 0) {
          println("Enter values of lambdas: ")
          layerLambdaOfC(j) = StdIn.readDouble()
          layerLambdaOfE(j) = StdIn.readDouble()
        }*/
        if (i != 0) {
          layerLambdaOfC(j) = 5
          layerLambdaOfE(j) = 5
        }
      }
      lambdaOfC(i) = layerLambdaOfC
      lambdaOfE(i) = layerLambdaOfE
    }
    Layers = layers.toSeq
    LambdaOfC = lambdaOfC.map(_.toSeq).toSeq
    LambdaOfE = lambdaOfE.map(_.toSeq).toSeq
  }

  def displayInput(samples: Seq[InputOutput], xrange: Int) = {
    val outputMatrix = DenseMatrix.zeros[Int](xrange,xrange)
    samples.foreach(x => outputMatrix((xrange*x.input(0)).toInt - 1,(xrange*x.input(1)).toInt - 1) = (x.output(0) + 1).toInt)
    print(s"\n$outputMatrix")
  }

  def readInput(xrange: Int, sampleType: String, file: Int): Stream[InputOutput] = {

    if(file == -2){
      println("How many samples")
      val count = StdIn.readInt()
      for (x <- MnistReader.trainDataset.examples.take(count)) yield new InputOutput(x._1, x._2)
    } else {
      val FilePrefix = "src/main/resources/sample/akka/"

      val fileName = if(file >= 0) {
        file
      } else {
        println("Enter the name of the data file")
        StdIn.readLine()
      }
      val source = io.Source.fromFile(s"$FilePrefix$sampleType/$fileName").getLines

      for (line <- source.toStream if !line.isEmpty)
        yield new InputOutput(DenseVector(line.split("=")(0).split(" ").map(_.toDouble/10)), DenseVector(line.split("=")(1).toDouble))
    }
  }

  def readTestingSamples(xrange: Int, sampleType: String, file: Int): Stream[InputOutput] = {

    if(file == -2){
      for (x <- MnistReader.testDataset.examples.take(1000)) yield new InputOutput(x._1, x._2)
    } else {
      val FilePrefix = "src/main/resources/sample/akka/"

      val fileName = if(file >= 0) {
        file
      } else {
        println("Enter the name of the data file")
        StdIn.readLine()
      }
      val source = io.Source.fromFile(s"$FilePrefix$sampleType/$fileName").getLines

      for (line <- source.toStream if !line.isEmpty)
        yield new InputOutput(DenseVector(line.split("=")(0).split(" ").map(_.toDouble/10)), DenseVector(line.split("=")(1).toDouble))
    }
  }

}
