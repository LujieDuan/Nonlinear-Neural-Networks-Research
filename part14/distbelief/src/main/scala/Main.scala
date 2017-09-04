/**
  * Created by LD on 2017-02-06.
  */
object Main extends App{

  println("MNIST with AKKA, shard size 10000")

  new GeneralMultilayer(784, 10, -2, None, Some(5.0), 10000, true)
}
