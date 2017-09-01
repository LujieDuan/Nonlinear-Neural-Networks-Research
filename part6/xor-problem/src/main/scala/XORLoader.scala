import breeze.linalg.DenseVector

object MnistLoader {

  /**
    * Load train/validation/testSet sets, with default rate of train:validation = 90:10
    * @return (train, validation, testSet)
    */
  def load(): DataSet = {
    val result = Seq[(DenseVector[Double], Double)]((DenseVector(0, 0), 0.0),
      (DenseVector(0, 1), 1.0),
      (DenseVector(1, 0), 1.0),
      (DenseVector(1, 1), 0.0))
    new DataSet(result, Seq.empty, Seq.empty)
  }
}
