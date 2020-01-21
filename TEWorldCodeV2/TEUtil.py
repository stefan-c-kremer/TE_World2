import math;
import random;

class Triangle:
  def __init__( self, pzero, pmax ):
    self.pzero = pzero;
    self.pmax = pmax;
    
  def sample( self ):
    return (self.pmax-self.pzero) * math.sqrt( random.random() ) + self.pzero; 

  def __repr__( self ):
    return "Triangle( %s, %s )" % ( repr(self.pzero),repr(self.pmax) );

class Flat:
  def __init__( self ):
    return;

  def sample( self ):
    return random.random();

  def __repr__( self ):
    return "Flat()";

    
class ProbabilitiesDontAddTo100( ValueError ):
  def __init__( self ):
    pass;
 
class ProbabilityTable:
  def __init__( self, *args ):
    self.table = [];
    args = list(args);

    prob = 0.0;
    while len(args)>0:
      prob += args.pop(0);
      value = args.pop(0);
      self.table.append( (prob, value ) );
    if len(args)!=0:
      raise BadArguments;
    if prob != 1.0:
      raise ProbabilitiesDontAddTo100;
      
  def generate( self ):
    rnd = random.random();
    for key, value in self.table:
      if rnd<key:
        return value;
    raise ProbabilitiesDontAddTo100;

  def __repr__( self ):
    result = "ProbabilityTable(";
    for prob, value in self.table:
      result += "%s,%s," % (repr(prob),'lambda?');
    result = result[:-1]+")";
    return result;

