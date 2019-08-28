import os
from threading import Thread
import hashlib
import json
import time
import glob


class externalfunc:
    
    def __init__(self , prog, names):
        self.call = prog
        self.N = names
        
    def __call__(self, X, folds):
        self.args = dict(zip(self.N,X))
        h = hashlib.md5(str(self.args).encode('utf-8')).hexdigest()
        com = '%s %s'% (self.call, ' '.join(['--%s %s'%(k,v) for (k,v) in self.args.items() ]))
        com += ' --hash %s'%h
        if folds: com +=' --folds %s'%folds
        com += ' > %s.log'%h
        #node = random.choice(['culture-plate-sm','imperium-sm','flere-imsaho-sm'])
        #com = 'ssh %s  '%node + com
        print( "Executing: ",com )
        ## run the command
        c = os.system( com )
        ## get the output
        try:
            r = json.loads(open('%s.json'%h).read())
            Y = r['result']
        except:
            print( "Failed on",com )
            Y = None
        return Y
        
        
class worker(Thread):
    
    def __init__(self,
                 X,
                 func,
                 folds=1):
        Thread.__init__(self)
        self.X = X
        self.used = False
        self.func = func
        self.folds = folds
        
    def run(self):
        self.Y = self.func(self.X, self.folds)
        
        
class manager:
    
    def __init__(self, n, skobj,
                 iterations, func, wait=10, folds = 1):
       
        self.n = n ## number of parallel processes
        self.sk = skobj ## the skoptimizer you created
        self.iterations = iterations
        self.folds = folds
        self.wait = wait
        self.func = func
        
    def run(self):
        
        print( 'Start the manager' )
        
        ## first collect all possible existing results
        for eh  in  glob.glob('*.json'):
            try:
                ehf = json.loads(open(eh).read())
                y = ehf['result']
                x = [ehf['params'][n] for n in self.func.N]
                print( "pre-fitting",x,y,"remove",eh,"to prevent this" )
                print( skop.__version__ )
                self.sk.tell( x,y )
            except:
                pass
        workers=[]
        it = 0
        asked = []
        while it< self.iterations:
            ## number of thread going
            n_on = sum([w.is_alive() for w in workers])
            if n_on< self.n:
                ## find all workers that were not used yet, and tell their value
                XYs = []
                for w in workers:
                    if (not w.used and not w.is_alive()):
                        if w.Y != None:
                            XYs.append((w.X,w.Y))
                        w.used = True
                    
                if XYs:
                    one_by_one= False
                    if one_by_one:
                        for xy in XYs:
                            print( "\t got",xy[1],"at",xy[0])
                            self.sk.tell(xy[0], xy[1])
                    else:
                        print( "\t got",len(XYs),"values" )
                        print( "\n".join(str(xy) for xy in XYs  ))
                        self.sk.tell( [xy[0] for xy in XYs], [xy[1] for xy in XYs])
                    asked = [] ## there will be new suggested values
                    print( len(self.sk.Xi))

                        
                ## spawn a new one, with proposed parameters
                if not asked:
                    asked = self.sk.ask(n_points = self.n)
                if asked:
                    par = asked.pop(-1)
                else:
                    print( "no value recommended" )
                it+=1
                print( "Starting a thread with",par,"%d/%d"%(it,self.iterations))
                workers.append( worker(
                    X=par ,
                    func=self.func,
                    folds = self.folds ))
                workers[-1].start()
                time.sleep(self.wait) ## do not start all at the same exact time
            else:
                ## threads are still running
                if self.wait:
                    #print n_on,"still running"
                    pass
                time.sleep(self.wait)
                
        while sum([w.is_alive() for w in workers]) > 0:
            time.sleep(self.wait)
        
                
def dummy_func_folded( X, folds=1):
    r = []
    for f in range( folds ):
        r.append( dummy_func( X, fold = f) )

    import numpy as np
    return np.mean( r )


def dummy_func( X , fold = None):
    import random
    print( "Providing a simple square as backup" )
    print( "fold",fold )
    Y = X[0]**2+X[1]**2 + random.random()*10
    return Y
                
            
def run_opt():
    
    print( 'Start optimization' )
    
    from skopt import Optimizer
    from skopt.learning import GaussianProcessRegressor
    from skopt.space import Real, Integer
    from skopt import gp_minimize
    from models.nlp_model import NLP
    import sys

    dim = NLP.get_skopt_dimensions()
    print( dim )
    
    names = []
    for var in dim:
        name = var.name
        print( name )
        names.append(name)
        
    
    folds = 1
    n_par = 2
    externalize = externalfunc(prog='python3 run_train_ex.py', names=names)
    
    run_for = 10

    use_func = externalize
    if len(sys.argv)>1:
        do = sys.argv[1]
        if do=='threaded':
            use_func = dummy_func_folded
        elif do=='external':
            use_func = externalize


    #dim = [Real(-20, 20) for i in range(n_par)]
    """
    start = time.mktime(time.gmtime())
    res = gp_minimize(
        func=lambda X : use_func(X, folds),
        dimensions=dim,
        n_calls = run_for,
        
    )

    print( "GPM best value",res.fun,"at",res.x )
    #print res
    print( "took",time.mktime(time.gmtime())-start,"[s]" )
    """
    
    o = Optimizer(
        n_initial_points =5,
        acq_func = 'gp_hedge',
        acq_optimizer='auto',
        base_estimator=GaussianProcessRegressor(alpha=0.0, copy_X_train=True,
                                                n_restarts_optimizer=2,
                                                noise='gaussian', normalize_y=True,
                                                optimizer='fmin_l_bfgs_b'),
        dimensions=dim,
    )

    m = manager(n = 2,
                skobj = o,
                iterations = run_for,
                func = use_func,
                wait= 10,
                folds = folds
    )
    start = time.mktime(time.gmtime())
    m.run()
    import numpy as np
    best = np.argmin( m.sk.yi)
    print( "Threaded GPM best value",m.sk.yi[best],"at",m.sk.Xi[best], )
    print( "took",time.mktime(time.gmtime())-start,"[s]" )
    

        
if __name__ == "__main__":
    
    run_opt()
        
        


            
