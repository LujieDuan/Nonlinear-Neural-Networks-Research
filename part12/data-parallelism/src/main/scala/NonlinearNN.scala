import Master.Start
import akka.actor.{ActorSystem, Props}

/**
  * Created by LD on 2017-05-24.
  */
object NonlinearNN {
  def main(args: Array[String]): Unit = {

    /**
      * A sample input is '0.01 20 100 10 784 10'
      */
    val learning_rate = args(0).toFloat
    val epoch = args(1).toInt
    val miniBatchSize = args(2).toInt
    val replicaCount = args(3).toInt
    val sizes = args.drop(4).map(_.toInt)
    val datasets = MnistLoader.load()

    val system = ActorSystem("NonlinearNets")

    val master = system.actorOf(Props(new Master(datasets, epoch, miniBatchSize, learning_rate, sizes, replicaCount)))

    master ! Start
  }
}
