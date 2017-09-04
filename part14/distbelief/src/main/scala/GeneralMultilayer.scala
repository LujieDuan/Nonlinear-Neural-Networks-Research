import Utils.{InputOutput, NetworkBuilder, SimulationType}
import actors.Master
import actors.Master.Start
import akka.actor.{ActorSystem, Props}

import scala.util.Random

/**
  * Created by LD on 2017-03-08.
  */
class GeneralMultilayer(dim: Int, outputDim: Int, file: Int, layers: Option[Seq[Int]], lambda: Option[Double], shardSize: Int, usingValidation: Boolean) {

  val random = new Random

/*  val possibleExamples = Seq(
    InputOutput(DenseVector(0.0, 0.0), 0.0)
    , InputOutput(DenseVector(0.0, 1.0), 1.0)
    , InputOutput(DenseVector(1.0, 0.0), 1.0)
    , InputOutput(DenseVector(1.0, 1.0), 0.0)
  )

  //generate 50000 training examples
  val trainingExamples = (1 to 50000).foldLeft(Seq[InputOutput]()) { (a, c) =>
    a :+ possibleExamples(random.nextInt(possibleExamples.size))
  }*/

  val simType = if (file == -2) SimulationType.Mnist else SimulationType.Grid

  val builder = new NetworkBuilder(dim, outputDim)

  val filePrefix = if(dim == 2) "2d" else "md"

  val trainingExamples = builder.readInput(10, filePrefix, file).toList

  val validationSamples = if(usingValidation) builder.readTestingSamples(10, filePrefix, file).toList else trainingExamples

  val system = ActorSystem("NetworkSystem")

  val master = if(layers.isDefined) {
    system.actorOf(Props(new Master(
      dataSet = trainingExamples,
      validationSet = Some(validationSamples),
      layers = layers.get,
      dataShardSize = shardSize,
      fixedLambda = lambda,
      lambdaOfC = builder.LambdaOfC,
      lambdaOfE = builder.LambdaOfE,
      simType = simType
    )), "Master")
  }
  else{
    builder.buildNetwork
    system.actorOf(Props(new Master(
      dataSet = trainingExamples,
      validationSet = Some(validationSamples),
      layers = builder.Layers,
      dataShardSize = shardSize,
      fixedLambda = lambda,
      lambdaOfC = builder.LambdaOfC,
      lambdaOfE = builder.LambdaOfE,
      simType = simType
    )), "Master")
  }

  if(dim == 2) builder.displayInput(trainingExamples, 10)
  master ! Start
}
