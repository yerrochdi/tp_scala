package com.sparkProject

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.types.{StructType, StructField, StringType, IntegerType}
import org.apache.spark.sql.functions._



object Job {

  def main(args: Array[String]): Unit = {

    // SparkSession configuration
    val spark = SparkSession
      .builder
      .appName("spark session TP_parisTech")
      .getOrCreate()

    val sc = spark.sparkContext

    import spark.implicits._


    /** ******************************************************************************
      *
      * TP 1
      *
      * - Set environment, InteliJ, submit jobs to Spark
      * - Load local unstructured data
      * - Word count , Map Reduce
      * *******************************************************************************/


    // ----------------- word count ------------------------
    /*
    val df_wordCount = sc.textFile("/home/errochdi/spark-2.0.0-bin-hadoop2.6/README.md")
      .flatMap{case (line: String) => line.split(" ")}
      .map{case (word: String) => (word, 1)}
      .reduceByKey{case (i: Int, j: Int) => i + j}
      .toDF("word", "count")

    df_wordCount.orderBy($"count".desc).show()
    */

    /** ******************************************************************************
      *
      * TP 2 : début du projet
      *
      * *******************************************************************************/
    /** 1-- a) Charger le fichier csv dans un dataFrame
      * (les premières lignes sont des explications sur les données et ne doivent pas être chargées).
      */
    val df = spark
      .read
      .option("header", "true") // Use first line of all files as header
      .option("inferSchema", "true") // Automatically infer data types
      .option("comment", "#")
      .csv("/home/errochdi/Cours_MS_BIG_DATA/Spark/cumulative.csv")

    /** 2-- b) Afficher le nombre de lignes et le nombre de colonnes dans le dataFrame */
    println("number of columns", df.columns.length)
    println("number of rows", df.count)

    /** 3-- d) Afficher le dataFrame sous forme de table. */
    df.show()

    /** Le dataFrame ayant un grand nombre de colonnes, afficher sous forme de table un sous-ensemble des colonnes (exemple les colonnes 10 à 20). **/
    val columns = df.columns.slice(10, 30)

    df.select(columns.map(col): _*).show(50)

    /** 3-- e) Afficher le schéma du dataFrame (nom des colonnes et le type des données contenues dans chacune d’elles)
      */
    df.printSchema()

    /** Pour une classification, l’équilibrage entre les différentes classes dans les données
      * d ’entraînement doit être contrôlé (et éventuellement corrigé). Afficher le nombre d’éléments de chaque classe (colonne koi_disposition).
      */
    println("Le bo")
    df.groupBy($"koi_disposition").count().show()


    /** ************************************************************************/
    /** Cleaning                                **/
    /** ************************************************************************/

    /** a ) Conserver uniquement les lignes qui nous intéressent pour le modèle (koi_disposition = CONFIRMED ou FALSE POSITIVE ) **/
    val df_cleaned = df.filter($"koi_disposition" === "CONFIRMED" || $"koi_disposition" === "FALSE POSITIVE")
    val columns_name = Seq("koi_disp_prov", "koi_disposition")
    df.select(columns_name.map(col): _*).show(50)

    /** b) Afficher le nombre d’éléments distincts dans la colonne “koi_eccen_err1”.
      * Certaines colonnes sont complètement vides: elles ne servent à rien pour l’entraînement du modèle.
      */
    df_cleaned.groupBy($"koi_eccen_err1").count().show()

    /** c) Enlever la colonne “koi_eccen_err1”. **/

    val df_cleaned2 = df_cleaned.drop($"koi_eccen_err1")

    /** d)
      * La liste des colonnes à enlever du dataFrame avec une seule commande:
      * La colonne “index" est en doublon avec “rowid”, on peut l’enlever sans perte d’information.
      * "kepid" est l’Id des planètes.
      * Les colonnes "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec” contiennent des informations provenant
      * d’autres mesures que la courbe de luminosité.
      * Si on garde ces colonnes le modèle est fiable dans 99.9% des cas ce qui
      * est suspect ! On enlève donc ces colonnes.
      * "koi_sparprov", "koi_trans_mod", "koi_datalink_dvr",
      * "koi_datalink_dvs", "koi_tce_delivname", "koi_parm_prov",
      * "koi_limbdark_mod", "koi_fittype", "koi_disp_prov",
      * "koi_comment", "kepoi_name", "kepler_name",
      * "koi_vet_date", "koi_pdisposition" ne sont pas essentielles.**/

    val columns_droped = Seq("index", "kepid", "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec",
      "koi_sparprov", "koi_trans_mod", "koi_datalink_dvr", "koi_datalink_dvs", "koi_tce_delivname",
      "koi_parm_prov", "koi_limbdark_mod", "koi_fittype", "koi_disp_prov", "koi_comment", "kepoi_name", "kepler_name",
      "koi_vet_date", "koi_pdisposition")

    val df_cleaned3 = df.drop(columns_droped: _*)

    val liste_colonnes = df_cleaned3.columns

    println("Nombre de colonne avant nettoyage" +
       + df_cleaned3.columns.length)

    var nb_elem: Long = 0L
    for( column_name <- liste_colonnes){
      println( "nom des colonnes: " + column_name );
      nb_elem = df_cleaned3.select(column_name).distinct().count()
      if(nb_elem <= 1){
        df_cleaned3.drop(column_name)
      }
    }
    println("Nombre de colonne après nettoyage" +
      + df_cleaned3.columns.length)

    /** e)
      * D’autres colonnes ne contiennent qu’une seule valeur (null, une string, un entier, ou autre).
      * Elles ne permettent donc pas de distinguer les deux classes qui nous intéressent.
      * Trouver le moyen de lister toutes ces colonnes non-valides ou bien toutes les colonnes valides, et ne conserver que ces dernières.
      */

    val useless_column = df_cleaned3.columns.filter{
      case (column:String) =>
      df_cleaned3.agg(countDistinct(column)).first().getLong(0) <= 1
          }

    println("useless_column")
    println(useless_column.length)

    val good_columns= df_cleaned3.columns
      .map(x => df_cleaned.select(x).distinct().count()<=1)
      .filter(_==true)


    var df_clean3 = df_cleaned3.columns.map(col => if(df_cleaned3.select(col).distinct().count() == 1) { df_cleaned3.drop(col) })

    println("nombre de colonnes après le 2nd nettoyage: " + df_clean3.length )

    /**f) Afficher des statistiques sur les colonnes du dataFrame,
      *  éventuellement pour quelques colonnes seulement pour la lisibilité du résultat.
      */
    df_cleaned3.describe("koi_impact", "koi_duration").show()

    /**g) Certaines cellules du dataFrame ne contiennent pas de valeur.
    Remplacer toutes les valeurs manquantes par zéro.
     **/
    val df_filled = df_cleaned3.na.fill(0.0)


    /**
      * 6 - Joindre deux dataFrames
      * */

    /**Comme en base de données relationnelles, on peut joindre deux dataFrames en un seul.
    Ici nous n’avons qu’une seule source de données et donc un seul dataFrame.
    Pour l’exercice suivant vous allez créer deux dataFrame à partir de celui obtenu à la fin de la section 4 :
      **/
    val df_labels = df_cleaned3.select("rowid", "koi_disposition")
    val df_features = df_cleaned3.drop("koi_disposition")

    val df_joined = df_features.join(df_labels, usingColumn = "rowid")

    /**
     * 8) Ajouter et manipuler des colonnes
    **/
    def udf_sum = udf((col1: Double, col2: Double) => col1 + col2)
    val df_newFeatures = df_cleaned3
      .withColumn("koi_ror_min", udf_sum($"koi_ror", $"koi_ror_err2"))
      .withColumn("koi_ror_max", $"koi_ror" + $"koi_ror_err1")

   /**Sauvegarder un dataFrame

Sauvegarder le dataFrame au format csv sur votre machine,
   assurez-vous d’avoir laissé le header du dataFrame (qui contient les noms des colonnes).**/

    df_newFeatures
      .coalesce(1)
      .write
      .mode("overwrite")
      .option("header", "true")
      .csv("/home/errochdi/Cours_MS_BIG_DATA/Spark/cleanedDataFrame.csv")
  }



}
