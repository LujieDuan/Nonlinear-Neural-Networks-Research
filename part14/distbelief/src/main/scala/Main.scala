import scala.io.StdIn

/**
  * Created by LD on 2017-02-06.
  */
object Main extends App{

  println("Select a simulation to run:\n" +
    "1.General Multilayer with AKKA, shard size 10\n" +
    "2.General Multilayer with AKKA, shard size 500\n" +
    "3.MNIST with AKKA, shard size 10000\n" +
    "")

  val simulation = StdIn.readInt()
  simulation match {
    case 1 => new GeneralMultilayer(2, 1, -1, None, None, 10, false)
    case 2 => new GeneralMultilayer(2, 1, -1, None, None, 500, false)
    case 3 => new GeneralMultilayer(784, 10, -2, None, Some(5.0), 10000, true)
  }
}
