/**
  * Created by LD on 2017-07-13.
  */
object Main {

  def main(args: Array[String]): Unit = {
    val fourNodes = Array(2, 1, 1)
    val fiveNodes = Array(2, 2, 1)
    val EPOCH = 100000
    val MINI_BATCH_SIZE = 1


    var LEARNING_RATE = 3.0

    new LinearNN(LEARNING_RATE, fourNodes, EPOCH, MINI_BATCH_SIZE).start()

    new LinearNN(LEARNING_RATE, fiveNodes, EPOCH, MINI_BATCH_SIZE).start()


    LEARNING_RATE = 0.5

    new NonlinearNN(LEARNING_RATE, fourNodes, EPOCH, MINI_BATCH_SIZE).start()

    new NonlinearNN(LEARNING_RATE, fiveNodes, EPOCH, MINI_BATCH_SIZE).start()


    LEARNING_RATE = 3.0

    new SingleTermNN(LEARNING_RATE, fourNodes, EPOCH, MINI_BATCH_SIZE).start()

    new SingleTermNN(LEARNING_RATE, fiveNodes, EPOCH, MINI_BATCH_SIZE).start()

  }

}
