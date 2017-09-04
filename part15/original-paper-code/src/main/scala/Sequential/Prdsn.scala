package Sequential

import java.io.{File, PrintWriter}

import Utils.{Compute, Control}

import scala.io.StdIn

/**
  * Created by LD on 2017-02-18.
  */
class Prdsn {

  /** Maximum number of iterations allowed */
  val MaxItr = 1000000

  /** Maximum number of input-output pairs */
  val Max = 500

  /** Error tolerance */
  val EP = 0.1

  /** Types */
  class Lnk {
    var wt: Double = 0
    var pwr: Double = 0
    var daw: Double = 0
    var nxt: Lnk = null
  }

  def newLnk: Lnk = {
    new this.Lnk
  }

  def LnkType = Array[Lnk](newLnk, newLnk, newLnk)

  def BigType = Array.ofDim[Double](3,3)
  def VerySmlr = new Array[Double](3)
  def MedInt = new Array[Int](Max)
  def MedRl = new Array[Double](Max)

  /** Variables */

  var seed :Int = 1
  var cur :Int = 0
  var num :Int = 0
  var count :Int = 0
  var counter :Int = 0
  var checker :Int = 0

  var minl :Double = 0
  var maxl :Double = 0
  var minm :Double = 0
  var maxm :Double = 0
  var dow :Double = 0
  var q :Double = 0           //Lambda
  var err :Double = 0
  var sec :Double = 0
  var prev :Double = 0
  var prod1 :Double = 0
  var prod2 :Double = 0
  var r :Double = 0           //Lambda

  var x = VerySmlr
  var dawp = VerySmlr
  var cst = VerySmlr
  var dawc = VerySmlr

  var d = MedInt              //Original Output
  var l = MedRl              //Original X
  var m = MedRl              //Original Y

  var t = LnkType

  var u = MedRl               //Scaled X
  var v = MedRl               //Scaled Y
  var tf = new StringBuilder  //Output Stream
  var ff = new String         //Input File Name


  /**
    * Produce random values in the interval [-1, 1]
    */
  def random: Double = {
    val modulus = 65536
    val multiplier = 25173
    val increment = 13849
    seed = (multiplier*seed+increment) % modulus
    ((seed.toDouble/modulus)-0.5)*2
  }

  /**
    * Fixes initial values for all learnable parameters, and initializes counters
    */
  def initialize: Unit = {

    checker = 0
    count = 0
    x(2) = x(0)
    err = 50

    for (i <- 0 to 2) {
      t(i).pwr = 1

      cst(i) = Compute.random1
      tf.++= (s"cst[$i]=${cst(i)}\n")

      t(i).wt = Compute.random1
      tf.++= (s"wt[$i]=${t(i).wt}\n")
    }

    println("Enter Lambdas:")
    q = StdIn.readDouble()
    r = StdIn.readDouble()
  }

  /**
    * Finds the output of unit i for input x
    */
  def sig(): Double = {
    prod1 = t(0).wt*Compute.pr(x(0), t(0).pwr)
    var tt = t(0)

    while (tt.nxt != null){
      tt = tt.nxt
      prod1 = prod1 + tt.wt*Compute.pr(x(0), tt.pwr)
    }

    prod1 = prod1 + cst(0)

    prod2 = prod2 + t(1).wt*Compute.pr(x(1), t(1).pwr)
    tt = t(1)

    while (tt.nxt != null){
      tt = tt.nxt
      prod2 = prod2 + tt.wt*Compute.pr(x(1), tt.pwr)
    }

    prod2 = prod2 + cst(1)

    var prod3 = t(2).wt*Compute.pr(x(2), t(2).pwr)
    tt = t(2)

    while (tt.nxt != null){
      tt = tt.nxt
      prod3 = prod3 + tt.wt*Compute.pr(x(2), tt.pwr)
    }

    prod3 = prod3 + cst(2)

    val prod = prod1 * prod2 + prod3

    1.toDouble / (1 + scala.math.exp(-5 * prod))

  }


  /**
    * runs the forward pass and assigns the result to 'sec'
    */
  def fwd = {
    sec = sig()
  }

  /**
    * modifies the values of weights using the values of computed derivatives
    */
  def change1 = {
    for (i <- 0 to 2){
      t(i).wt = t(i).wt - q * t(i).daw
      var tt = t(i)

      while (tt.nxt != null){
        tt = tt.nxt
        tt.wt = tt.wt - q * tt.daw
      }
      cst(i) = cst(i) - q * dawc(i)
    }
  }

  /**
    * modifies the exponent values and adds new polynomial term if required
    */
  def changep1 = {
    for (i <- 0 to 2) {
      if (t(i).pwr - r * dawp(i) >= 0) {
        t(i).pwr = t(i).pwr - r * dawp(i)
      }

      if (t(i).nxt != null){
        if (t(i).pwr - t(i).nxt.pwr > 2){
          print('.')
          val newt = newLnk
          newt.pwr = t(i).nxt.pwr + 1
          newt.nxt = t(i).nxt
          newt.wt = 0
          newt.daw = 0
          t(i).nxt = newt
        }
      }
      else if (t(i).pwr > 2) {
        print('.')
        val newt = newLnk
        newt.pwr = 1
        newt.nxt = null
        newt.wt = 0
        newt.daw = 0
        t(i).nxt = newt
      }
    }
  }

  /**
    * backward pass of the back propagation algorithm... Computes all derivatives with
    * respect to the error and modifies parameter values
    */
  def backtrack = {
    dow = (sec - d(cur))*sec*(1-sec)
    t(0).daw = dow*Compute.pr(x(0), t(0).pwr)* prod2
    var tt = t(0)

    while (tt.nxt != null) {
      tt = tt.nxt
      tt.daw = dow*Compute.pr(x(0), tt.pwr)* prod2
    }
    dawc(0) = dow * prod2

    t(1).daw = dow*Compute.pr(x(1), t(1).pwr) * prod1
    tt = t(1)

    while (tt.nxt != null) {
      tt = tt.nxt
      tt.daw = dow*Compute.pr(x(1), tt.pwr) * prod1
    }
    dawc(1) = dow * prod1

    t(2).daw = dow*Compute.pr(x(0), t(2).pwr)
    tt = t(2)

    while (tt.nxt != null) {
      tt = tt.nxt
      tt.daw = dow*Compute.pr(x(0), tt.pwr)
    }
    dawc(2) = dow

    change1

    dawp(0) = dow * t(0).wt * Compute.pr(x(0), t(0).pwr) * Compute.lan(x(0)) * prod2
    dawp(1) = dow * t(1).wt * Compute.pr(x(1), t(1).pwr) * Compute.lan(x(1)) * prod1
    dawp(2) = dow * t(2).wt * Compute.pr(x(0), t(2).pwr) * Compute.lan(x(0))

    changep1
  }


  /**
    * returns the value of error
    */
  def error: Double = {
    prev = err
    err = scala.math.abs(d(cur) - sec)
    err
  }

  /**
    * returns 'true' if the error is more than 'ep'. Otherwise returns 'false'
    */
  def correct: Boolean = if (error <= EP) true else false

  /**
    * runs iterations until either all the input-output pairs have been learnt or the highest
    * number of iterations has been reached
    */
  def all = {
    var prevchk = 0
    cur = 0
    x(0) = u(cur)
    x(1) = v(cur)
    x(2) = x(0)
    fwd

    do {
      if (!correct) {
        backtrack
        checker = 0
        cur = cur % num + 1
        x(0) = u(cur)
        x(1) = v(cur)
        x(2) = x(0)
      }
      else {
        checker += 1
        cur = cur % num + 1
        x(0) = u(cur)
        x(1) = v(cur)
        x(2) = x(0)
      }

      fwd
      count += 1

      if (checker > prevchk) {
        println(checker)
        prevchk += 1
      }
    } while (checker < num && count <= MaxItr)
  }


  /**
    * writes the learnt functions to a file
    */
  def wrtres = {
    for (i <- 0 to 2) {
      tf.++=(s"synapse $i\n\n")
      tf.++=(s"[(${t(i).wt}) X$i ^ ${t(i).pwr}]\n")
      var tt = t(i)

      while (tt.nxt != null) {
        tt = tt.nxt
        tf.++=(s" + [(${tt.wt}) X $i ^ ${tt.pwr.toInt}]\n")
      }

      tf.++=(s" + ${cst(i)}")
      tf.++=("\n")
    }
  }

  /**
    * reads the value of 'seed' to be used for the random number generator from a file
    */
  def globini = {
    Control.using(io.Source.fromFile("seed")) { source => {
      for (line <- source.getLines) {
        seed = line.mkString.toInt
      }
    }}
  }

  /**
    * Calls 'wrtres' to write the learnt parameters to the output file.
    * Output the number of iterations required if a solution was found and informs if number
    * of iterations exceeded the limit
    */
  def respond = {
    if (count > MaxItr) {
      wrtres
      tf.++=("Number of iterations exceeded!\n")
    }
    else if (checker >= num) {
      wrtres
      tf.++=("\n")
      tf.++=(s"Got it!... In $count iterations\n")
      tf.++=("\n")
      tf.++=(s"Values of lambdas = $q $r\n")
      tf.++=("\n")
    }
  }

  /**
    * reads input-output pairs from the input file
    */
  def readinp = {
    println("Enter the data file number:")
    ff = StdIn.readLine()
    var iterator = 0
    Control.using(io.Source.fromFile(s"src/main/resources/sample-$ff" )) { source => {
      val inputIntegerList = source.mkString.split(" ").map(_.toInt)
      num = inputIntegerList(iterator)
      iterator += 1
      l(0) = inputIntegerList(iterator)
      iterator += 1
      m(0) = inputIntegerList(iterator)
      iterator += 1
      d(0) = inputIntegerList(iterator)
      iterator += 1
      minl = l(0)
      maxl = l(0)
      minm = m(0)
      maxm = m(0)

      for (i <- 1 until num) {
        l(i) = inputIntegerList(iterator)
        iterator += 1
        m(i) = inputIntegerList(iterator)
        iterator += 1
        d(i) = inputIntegerList(iterator)
        iterator += 1

        if (l(i) < minl) {
          minl = l(i)
        }
        else if (l(i) > maxl) {
          maxl = l(i)
        }

        if (m(i) < minm) {
          minm = m(i)
        }
        else if (m(i) > maxm) {
          maxm  = m(i)
        }
      }
    }}

    if (maxl == minl) maxl += 1
    if (maxm == minm) maxm += 1

    for (i <- 0 until num) {
      u(i) = (l(i) - minl) / (maxl - minl)
      v(i) = (m(i) - minm) / (maxm - minm)
    }
  }

  /**
    * creates a bitmap for the learnt function
    */
  def display = {
    for (i <- minl.toInt to maxl.toInt) {
      for (j <- minm.toInt to maxm.toInt) {
        x(0) = (i - minl) / (maxl - minl)
        x(1) = (j - minm) / (maxm - minm)
        x(2) = x(0)
        fwd
        tf ++= (sec + 0.5).toInt.toString
      }
      tf ++= "\n"
    }
    val pw = new PrintWriter(new File(s"src/main/resources/prdsn-result-$ff" ))
    pw.write(tf.toString())
    pw.close()
    println(tf.toString())
    showInput
    println(s"Write Result to file src/main/resources/prdsn-result-$ff")
  }

  /**
    * saves value of seed
    */
  def saveseed = {
    val pw = new PrintWriter(new File("seed" ))
    pw.write(seed)
    pw.close()
  }

  /**
    * main program
    */
  def MainRouting = {
    //globini // loads seed value for 'random'
    readinp // reads input-output pairs
    val start = System.currentTimeMillis
    initialize // initializes the network
    all //runs iterations
    respond // outputs learnt functions
    display // outputs bitmap for learnt function
    val totalTime = System.currentTimeMillis - start
    //saveseed // saves seed value
    println("Elapsed time: %1d ms".format(totalTime))
  }

  /**
    * Show input in matrix form
    */
  def showInput: Unit = {
    val inputArray = Array.fill[Int](maxl.toInt - minl.toInt + 1, maxm.toInt - minm.toInt + 1)(-1)
    for (i <- 0 until num){
      inputArray(l(i).toInt - 1)(m(i).toInt - 1) = d(i)
    }
    inputArray foreach { row => row foreach {element => if (element < 0) print(" ") else print(element)}; println }
  }

}
