import utils as ut

def get_dirs( timerange ):
    print timerange
    dirs = ut.getDirs( timerange )
    print dirs
    return dirs

def load_data( sc, timerange ):
    
    if timerange[1] < 20171010:
        schema_file = 'hdfs:///cms/wmarchive/avro/schemas/current.avsc.20161215'
    else:
        schema_file = 'hdfs:///cms/wmarchive/avro/schema.avsc'
    rdd = sc.textFile(schema_file, 1).collect()
    
    # define input avro schema, the rdd is a list of lines (sc.textFile similar to readlines)
    avsc = reduce(lambda x, y: x + y, rdd) # merge all entries from rdd list
    schema = ''.join(avsc.split()) # remove spaces in avsc map
    conf = {"avro.schema.input.key": schema}
    
    # define newAPIHadoopFile parameters, java classes
    aformat="org.apache.avro.mapreduce.AvroKeyInputFormat"
    akey="org.apache.avro.mapred.AvroKey"
    awrite="org.apache.hadoop.io.NullWritable"
    aconv="org.apache.spark.examples.pythonconverters.AvroWrapperToJavaConverter"
    
    data_path = get_dirs( timerange )
    # load data from HDFS
    if  isinstance(data_path, list):
        avro_rdd = sc.union([sc.newAPIHadoopFile(f, aformat, akey, awrite, aconv, conf=conf) for f in data_path])
    else:
        avro_rdd = sc.newAPIHadoopFile(data_path, aformat, akey, awrite, aconv, conf=conf)
    
    return avro_rdd
        

        
    
    
    
    
    
    
    
    
    
    
    
        
        