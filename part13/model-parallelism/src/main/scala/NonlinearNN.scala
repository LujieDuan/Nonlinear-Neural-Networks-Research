import Master.Start
import akka.actor.{ActorSystem, Props}

/**
  * Created by LD on 2017-05-24.
  */
object NonlinearNN {

  var sizes: Seq[Int] = _

  def main(args: Array[String]): Unit = {

    /**
      * A sample input: '3 784 30 10'
      */
    val learning_rate = args(0).toDouble
    sizes = args.drop(1).map(_.toInt)
    println(s"Actor Based Single Model Nonlinear Neural Nets:${sizes.toString()}")
    val training = MnistLoader.trainDataset.examples.toArray
    val test = MnistLoader.testDataset.examples.toArray
    val epoch = 10
    val mini_batch_size = 10
    println(s"Batch Size: $mini_batch_size. Learning Rate: $learning_rate. Iterations: $epoch.")

    val system = ActorSystem("NonlinearNets")

    val master = system.actorOf(Props(new Master(training, epoch, mini_batch_size, learning_rate, test, sizes)))

    master ! Start

  }

}
