/**
  * Created by LD on 2017-07-13.
  */
object Main {

  def main(args: Array[String]): Unit = {
    val LEARNING_RATE = 3.0
    val EPOCH = 1
    val MINI_BATCH_SIZE = 100
    val layer_structures = args.map(_.toInt)

    import Common.{sigmoid, sigmoid_prime}
    new NonlinearNN(LEARNING_RATE, layer_structures, EPOCH, MINI_BATCH_SIZE, sigmoid, sigmoid_prime).start()


    import Common.{gaussian, gaussian_prime}
    new NonlinearNN(LEARNING_RATE, layer_structures, EPOCH, MINI_BATCH_SIZE, gaussian, gaussian_prime).start()


    import Common.{relu, relu_prime}
    new NonlinearNN(LEARNING_RATE, layer_structures, EPOCH, MINI_BATCH_SIZE, relu, relu_prime).start()

  }

}
