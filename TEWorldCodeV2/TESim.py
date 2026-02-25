import os;
import sys;
import time;
import gzip;
import random;
import bisect;
import glob

################################################################################
# Launch code
################################################################################


import TEUtil;

sys.path[0] = ''; # replace directory where this program resides with the experiment's
                    # directory

import parameters;


################################################################################


################################################################################
      
JUNK = "Junk";
 
def mean( items ):
  return sum(items)/len(items);
  
  
def Quartiles( key, numbers ):
  """
  This function identifies the 0, 25, 50, 75 and 100 percentile elements in 
  the list of number (which must be sorted).  The elements are returned in 
  a dictionary whose keys are given by the key (three letter acroymn) with
  the percentile string substituted.
  If numbers contains less than 5 elements it retuns a dictionary with all values
  equal to int(0).
  """

  numbers.sort();
  l = (len(numbers)-1)/4.0;
  if l < 0:
    return { key%0   :0,
             key%25  :0,
             key%50  :0,
             key%75  :0,
             key%100 :0 };
  quartiles = { key % (q*25,): numbers[int(q*l)] for q in range(0,5) };

  return quartiles;
  
################################################################################

def output( keyword, message ):
  """
  Print outputs.
  """
  if not parameters.output.has_key(keyword) or parameters.output[keyword]:
    print "[%s]: %s" % (keyword, message);
  
################################################################################
  
class ElementDestroyed( Exception ):
  """
  This is an exception which occurs when something tries to insert into an 
  existing element.
  E.g. a TE jumps into a gene, thereby destroying the gene.
  The exception is raised INSTEAD of performing the insertion.  If you want
  to complete the insertion (destroying the original element) you can call
  Chromosome.insert_anyway
  """
  def __init__( self, element, whats_there ):
    self.element = element;
    self.whats_there = whats_there;
    
################################################################################
  
class Element:
  """
  This represents an object of interest in the chromosome.
  I.e. a TE or a gene.
  It has a host chromosome (class Chromosome), a start (int) an 
  end (int), and a length (int).
  Subclasses may have additional attributes.
  """
  
  def __init__( self, length, start ):
    self.length = length;
    if type(start)!=int:
      print start;
      raise "Error:  TE start location is not an integer";
    self.start = start;
    self.end = self.start + self.length;

      
  def __repr__( self ):
    return "%s( %s, %s )" % (self.__class__.__name__, \
            repr(self.length), \
            repr(self.start) );
            
  ########################## Various comparator methods ########################
  # These redefine existing comparator methods within Python for the Element class instances
  
  def __lt__( self, other ):
    return self.start < other.start;

  def __le__( self, other ):
    return self.start <= other.start;
    
  def __eq__( self, other ):
    return self.start == other.start;
    
  def __ne__( self, other ):
    if isinstance( other, Element ):
      return self.start != other.start;
    else: # comparing element to something that's not an element
      return True;  # can't be equal - != is True
    
  def __gt__( self, other ):
    return self.start > other.start;
    
  def __ge__( self, other ):
    return self.start >= other.start;
    
################################################################################
            
class ProkGene1(Element):
  """
  Simple model of a prokaryotic gene which has no 
  introns and a fixed gene length of 1600 BPs.
  """
  length = parameters.Gene_length;
  
  def __init__( self, start ):
    Element.__init__( self, self.length, start );
    
  def copy( self ):
    return ProkGene1( self.start );

  def __repr__( self ):
    return "ProkGene1( %s )" % ( repr(self.start), );

################################################################################

class SelectiveInsertTE(Element):
  """
  A transposable element that has a preferred insertion  distribution 
  centred at self.mean, with width self.std and a length of 300 BPs.
  """
  
  def __init__( self, start, dead=False, autonomous=True):
    self.dead = dead;
    Element.__init__( self, parameters.TE_length, start );
    self.autonomous = autonomous
    
  def jump( self ):
    """
    This produces 0-3 copies of the TE and inserts them into the same chromosome.
    """
    jump_effects = { 'TEDEATH':  0, # death
                     'COLLISIO': 0, # collision
                     'TOTAL_JU': 0, # total
                     'LETHAL_J': 0, # lethal
                     'DELETE_J': 0,  # deleterious
                     'NEUTRA_J': 0, # neutral
                     'BENEFI_J': 0 };	# beneficial

    if self not in self.chromosome.elements:	
					# this happens if another element jumped into this element
      return jump_effects;

    if random.random() < parameters.TE_death_rate:    # mutation or host defenses
      self.dead = True;
      jump_effects['TEDEATH'] += 1;
      
    # Should return if the TE is now dead or non-autonomous
    # The rest of the functionality is not relevant to non-autonomous elements
    # There should be an early return, if the function has not returned yet, in this case
    if self.dead and not self.autonomous:
      return jump_effects;

    if parameters.TE_excision_rate==0.0:	# assume retro-transposon
      # orignal element survives and creates progeny
      progeny = parameters.TE_progeny.generate();
    
    # assume DNA transposon   
    elif random.random() < parameters.TE_excision_rate:
      # excise original element
      self.chromosome.excise( self );
      progeny = parameters.TE_progeny.generate();

    # DNA transposon but did not excise
    else:
      progeny = 0;

    # Iterating through progeny count (create offspring)
    for _ in range(progeny):
      jump_effects['TOTAL_JU'] += 1;
      try:
        new_te = self.birth()
        kidnapping_probability = self.chromosome.get_kidnapping_probability()
        
        # Simulate kidnapping on newly created autonomous TE, to convert it to the non-autonomous
        if random.random() < kidnapping_probability:
          new_te.autonomous = False
        
        jump_effects['COLLISIO'] += self.chromosome.insert( new_te );	# record TE collisions
      except ElementDestroyed, e:
        output( "SPLAT", "SPLAT!" );
        ind = self.chromosome.host;	# host is None
        new_fitness = parameters.Insertion_effect.generate()(ind.fitness);
        output( "SPLAT FITNESS", new_fitness );

        if new_fitness == 0.0:
          jump_effects['LETHAL_J'] += 1;	# jump death
        elif new_fitness < ind.fitness:
          jump_effects['DELETE_J'] += 1;	# deleterious
        elif new_fitness == ind.fitness:
          jump_effects['NEUTRA_J'] += 1;	# neutral
        else:
          jump_effects['BENEFI_J'] += 1;	# beneficial


        if ind.fitness > 0.0:	# can't change fitness of dead host
          ind.fitness = new_fitness;

        if ind.fitness < 0.0:
          ind.fitness = 0.0;

    return jump_effects; # return the number of TEs that are created and jump
    
  def birth( self ):
    """
    Create a copy of this TE.
    The copy will have a starting location in the host chromosome based
    on the probability distribution of the parent TE and will have a
    mutated probability distribution of its own (to be applied to its
    children).
    """
    # compute start location of new TE based on probabilty distro
    start = int(parameters.TE_Insertion_Distribution.sample()*self.chromosome.length);
    baby = SelectiveInsertTE( start=start, autonomous=self.autonomous)
    baby.chromosome = self.chromosome;
        
    return baby;
  
  def copy( self ):
    te_copy = SelectiveInsertTE( self.start );    
    te_copy.dead = self.dead;
    te_copy.autonomous = self.autonomous
    te_copy.chromosome = self.chromosome;
    return te_copy;

  def __repr__( self ):
    return "SelectiveInsertTE( %s, %s, %s )" % ( self.start, self.dead, self.autonomous );

################################################################################
     
      
class Chromosome:
  """
  Chromosome to hold Elements and junk.
  `elements` is a list of the elements in the chromosome
  `length` is the total length of the chromosome including elements and junk.
  """
  def __init__( self, length=parameters.Junk_BP, elements=None ):
    """
    Create a chromosome of given length of `junk' DNA.
    """
    if type( length ) not in [ int, float ]:
      raise Exception, repr(length);

    self.length = length;
    if elements==None:
      self.elements = []; # chromosome containing nothing but junk
    else:
      self.elements = elements; 
      for element in self.elements:
        element.chromosome = self;
   
  def place( self, element ):
    """
    Place gene into the Chromosome, taking the place of Junk without increasing
    the Chromosome length.
    """
    whats_there = self[ element.start ];
    if whats_there != JUNK:         # Nothing
      if isinstance( whats_there, SelectiveInsertTE ):
        self.remove( whats_there );
      # Raise ElementDestroyed error, if a gene already exists in the location
      else:
        raise ElementDestroyed( element, whats_there );

    # Insert the new element in sorted order into the list
    # of elements.
    bisect.insort_left( self.elements, element );

 
  def insert( self, element ):
    """
    Insert element into the Chromosome, moving stuff behind it back and
    increasing the Chromosome length.
    Return 1 if the element inserts into another element, 0 otherwise.
    """

    collision = 0;

    # figure out what's there
    whats_there = self[ element.start ];
    if whats_there != JUNK:         # Nothing
      if isinstance( whats_there, SelectiveInsertTE ):
        self.remove( whats_there );
        collision = 1;
      else:
        if element.start != whats_there.start:
          self.insert_anyway( element );
          element.chromosome = self;	# needs to be here!
          raise ElementDestroyed( element, whats_there );

    # inserting into junk      
    self.insert_anyway( element );

    element.chromosome = self;    
    return collision;

  def insert_anyway( self, element ):
    """
    Place element at its desired location without
    regard for whether that location is inside another
    element.
    
    Push elements that are later further back.
    Note the exception handler for ElementDestroyed
    should remove(...) the original element if it was hit.
    """
    
    # Adjust start and end of all elements that come after
    # the newly inserted on.
    for old_element in self.elements:
      # could make this more efficient via binary search
      if old_element.start >= element.start:
        new_start = old_element.start + element.length;
        old_element.start = new_start;
        old_element.end += element.length;
    
    # Insert the new element in sorted order into the list
    # of elements.
    bisect.insort_left( self.elements, element );
    
    # update the chromosome length
    self.length += element.length;
    
  def excise( self, element ):
    # shift following element over
    for old_element in self.elements:
      if old_element.start > element.start:
        new_start = old_element.start - element.length;
        old_element.start = new_start;
        old_element.end -= element.length;

    try:
      self.elements.remove(element);	# removed the element
    except ValueError, e:
      print ">>>325>>>", element;
      raise;

    self.length -= element.length;  	# update chromosome length

  def __getitem__( self, index ):
    """
    Returns the element at the given index, or Junk if no element exists
    at the index.
    """
  
    # could make this more efficient via binary search
    for element in self.elements:
      if element.start <= index < element.end: # start of element in region
        return element;

    return JUNK;
        
  def remove( self, item ):
    """
    Deletes the item from the list of elements but leaves the base pairs there
    effectively turning the DNA into Junk.
    """
    self.elements.remove( item );
    
  def __repr__( self ):
    """
    Return a representation of this object.
    """
    return "%s( %s, %s )" % \
            ( self.__class__.__name__, 
              repr(self.length), 
              repr(self.elements ) );
            
  def genes( self ):
    """
    Return a list of only ProkGene1 class elements.
    """
    return [ element for element in self.elements \
                     if isinstance(element,ProkGene1)];
    
  def TEs( self, live=True, dead=True, autonomous=None):
    """
    Return a list of only SelectiveInsertTE class elements that match the argument values provided.
    """
    filtered_tes = [ element for element in self.elements \
                     if isinstance(element, SelectiveInsertTE) and
                        (element.dead == dead or element.dead != live) ]
    
    if autonomous is None:
      return filtered_tes
    
    return [te for te in filtered_tes if te.autonomous == autonomous]
    
  def junk( self ):
    """
    Return the number of "junk" BPs.
    """
    return self.length - sum( [ element.length for element in self.elements ]);
    
  def copy( self, host ):
    """
    Return a copy of this chromosome with elements that have the same 
    coordinates.
    """
    # create deep copy of this chromosome
    
    result = self.__class__( length=self.length );  # create new chromosome instance

    
    for e in self.elements:
      ne = e.copy();  # create new element with identical properties
      ne.chromosome = result;
      result.elements.append( ne );
    return result;
   
################################################################################
 

        
################################################################################
        
class TestChromosome2(Chromosome):
  """
  Variation on Chromosome where genes are not evenly distributed,
  instead distributed in a probability distribution.
  """
  # number of genes to start with
  gene_no = parameters.Initial_genes;
  TE_no = parameters.Initial_TEs;           # number of TEs to start with
  length = parameters.Junk_BP
  
  def add_elements( self, genes=gene_no, TEs=TE_no ):
    while len( self.genes() ) < genes:
      try:
        start = self.genestart();      
        gene = ProkGene1(start = start);
        self.insert( gene );
      except ElementDestroyed, e: # most recent gene overwrote another
        if hasattr( parameters, "Append_gene" ) and parameters.Append_gene:
          gene.start = e.whats_there.end;	# move gene to end of previous
          gene.end = gene.start + gene.length;
        else:
          continue;  # while loop (try again)
    
    while len( self.TEs() ) < TEs:      
      try:
        start = self.testart();
        te = SelectiveInsertTE(start=start, dead=False, autonomous=True)
        self.insert(te);  # insert single TE instance
      except ElementDestroyed, e: # most recent if TE is in a gene
        continue;  # while loop (try again)
      output( "TE INIT", "%s" % te );
 
  def genestart( self ):
    """
    Return starting position for a gene.
    """
    return int( parameters.Gene_Insertion_Distribution.sample() * \
                (self.length) );
                #(self.length-parameters.Gene_length) );
    
  def testart( self ):
    return int(parameters.TE_Insertion_Distribution.sample()*self.length);
  
  def get_kidnapping_probability(self):
    """
    Computes the probability that a SINE will kidnap a LINE's retransposition protein, so that it can reproduce.
    """
    # Obtain the number of parasitic TEs, to obtain a ratio to be used for kidnapping probability
    n_autonomous = len(self.TEs(live=True, dead=False, autonomous=True))
    n_non_autonomous = len(self.TEs(live=True, dead=False, autonomous=False))
    
    return parameters.Kidnapping_Frequency(n_autonomous, n_non_autonomous)
    
  def jump( self ):
    jump_effects = { 'TEDEATH':  0, 
                     'COLLISIO': 0, 
                     'TOTAL_JU': 0, 
                     'LETHAL_J': 0, 
                     'DELETE_J': 0, 
                     'NEUTRA_J': 0, 
                     'BENEFI_J': 0 };	

    for te in self.TEs(live=True, dead=False):
      te_jump_effects = te.jump();
      jump_effects = { key: value + te_jump_effects[key] \
                                      for key,value in jump_effects.items() };

    return jump_effects;
 
################################################################################
    
class Species:
  """
  Represents species.
  """
  def __init__( self, cells, chromosomes ):
    self.cells = cells;
    self.chromosomes = chromosomes;

  def __repr__( self ):
    # chromosomes is a list of classes so we have to fudge the repr
    repr_chromosomes = "[ "+", ".join( [ chromosome.__name__ for chromosome in self.chromosomes ] ) + " ]";

    return "Species( %s, %s )" % ( repr(self.cells), repr_chromosomes );
    
################################################################################
    
class Host:
  """
  This class represents an individual host.
  """
  
  # class variables

  def __init__( self, species, chromosome=None, fitness=parameters.Host_start_fitness ):
    """
    Initialize this host.
    """
    self.species = species;
    if chromosome is None:	# no chomosomes provided; create them
      self.chromosome = [ chromosome() \
                            for chromosome in self.species.chromosomes ];
    else:
      self.chromosome = chromosome;	# use provided chromosomes

    for chromosome in self.chromosome:	# tell chromosomes who their host is
      chromosome.host = self;

    self.fitness = fitness;

  def jump_and_mutate( self ):
    jump_effects = { 'TEDEATH':  0, 
                     'COLLISIO': 0, 
                     'TOTAL_JU': 0, 
                     'LETHAL_J': 0, 
                     'DELETE_J': 0, 
                     'NEUTRA_J': 0, 
                     'BENEFI_J': 0 };	

    for chromosome in self.chromosome:
      chromosome_jump_effects = chromosome.jump();
      jump_effects = { key: value + chromosome_jump_effects[key] \
                                       for key,value in jump_effects.items() };
      # apply jump to each chromosome

    if random.random() < parameters.Host_mutation_rate:
      if self.fitness > 0.0:
        self.fitness = parameters.Host_mutation.generate()( self.fitness ); 
                                    # compute host fitness

    if self.fitness < 0.0:
      self.fitness = 0.0;

    return jump_effects;

    
  def clone( self ):
    """
    Clone host organism.  Returns a new organism with its own chromosome and
    genes that looks just like the original.
    """

    host = Host( self.species, [], self.fitness );

    # Create new array of new chromosomes
    chromosomes = [c.copy(host) for c in self.chromosome]; 
    host.chromosome = chromosomes;

    for chromosome in host.chromosome:
      chromosome.host = host;

    return host;
    
  def __repr__( self ):
    return "Host( %s, %s, %s )" % ( repr(self.species), repr(self.chromosome), repr(self.fitness) );

################################################################################
        
class Population:
  """
  This class represents a host population of individuals.
  """
    
  def __init__( self, capacity, species, individual=None, generation_no=0 ):
    self.capacity = capacity;
    self.species = species;

   
    if individual is None: 	# no individuals supplied, create them
      host = Host( species );   # generate a single individual
      host.chromosome[0].add_elements();
      
      autonomous_count = 0
      non_autonomous_count = 0
      for te in host.chromosome[0].TEs():
        if te.autonomous:
          autonomous_count += 1
        else:
          non_autonomous_count += 1
      
      # Clone hosts n times
      self.individual = [ host.clone() for _ in range(0,capacity) ]; 
      
      # After cloning the hosts we will distribute the non-autonomous TEs evenly
      non_autonomous_even_count = int(parameters.Total_NAut_TE / parameters.Carrying_capacity)
      non_autonomous_remainder = parameters.Total_NAut_TE % parameters.Carrying_capacity
      
      for host in self.individual:
        non_autonomous_to_add  = non_autonomous_even_count
        
        # Add remainder until there is no more left
        if non_autonomous_remainder > 0:
          non_autonomous_to_add += 1
          non_autonomous_remainder -= 1
          
        chromosome = host.chromosome[0]
        # Create non-autonomous TEs and insert them
        n_autonomous_tes = len(chromosome.TEs())
        
        # Ensuring that non_autonomous_to_add number of TEs were added correctly to the host
        while (len(chromosome.TEs()) - n_autonomous_tes) < non_autonomous_to_add:
          try:
            start = chromosome.testart()
            
            te = SelectiveInsertTE(start=start, dead=False, autonomous=False)
            chromosome.insert(te)
          except ElementDestroyed:
            # Try again
            continue
      
    else:
      self.individual = individual;
        
    self.generation_no = generation_no;
      
  def replication( self ):
    # clone a copy of each individual add them to the end of the individual 
    # list
    # this doubles the number of individuals in the world
    for dup in range(parameters.Host_reproduction_rate):
      self.individual += [ i.clone() for i in self.individual ];
  
  def jump_and_mutate( self ):
    jump_effects = { 'TEDEATH':  0, 
                     'COLLISIO': 0, 
                     'TOTAL_JU': 0, 
                     'LETHAL_J': 0, 
                     'DELETE_J': 0, 
                     'NEUTRA_J': 0, 
                     'BENEFI_J': 0 };	

    for ind,i in zip(self.individual,range(len(self.individual))):
      individual_jump_effects = ind.jump_and_mutate();
      jump_effects = { key: value + individual_jump_effects[key] \
                                       for key,value in jump_effects.items() };

    return jump_effects;
          
    
  def selection_and_drift( self ):
    total_fitness = sum( [ i.fitness for i in self.individual ]);
    if total_fitness > 0.0:
      new_population = [ i for i in self.individual 
                        if random.random() < parameters.Host_survival_rate( i.fitness/total_fitness ) ];
    else:
      new_population = [];

    self.individual = new_population;

        
  def __getitem__( self, index ):
    return self.individual[index];
    
  def generation( self ):
    """
    Simulate one generation of the population.
    """

    livetes =[ (len(individual.chromosome[0].TEs(live=True,dead=False))) 
                                        for individual in self.individual];
    self.replication();   # first replicate individuals (double population)

    livetes =[ (len(individual.chromosome[0].TEs(live=True,dead=False))) 
                                        for individual in self.individual];
    te_effects = \
         self.jump_and_mutate();       # apply jumping and mutation

    livetes =[ (len(individual.chromosome[0].TEs(live=True,dead=False))) 
                                        for individual in self.individual];

    self.selection_and_drift();   # apply selection and drift 
                                  # (population goes back down)

    livetes =[ (len(individual.chromosome[0].TEs(live=True,dead=False))) 
                                        for individual in self.individual];
    self.generation_no += 1;
    return te_effects;

     
  def __str__( self ):
    result = "Population size: %d\n" % (len(self.individual));
    

    
    return result;

  def __repr__( self ):
    return "Population( %s, %s, %s, %s )" % ( repr(self.capacity),
                                              repr(self.species),
                                              repr(self.individual),
                                              repr(self.generation_no), );
    
################################################################################
    
def test_triangle():
  print "pbins = triangle(100/3,100)";
  pbins = { bin:0 for bin in range(20) };
  for i in range(0,100000):
    x = triangle( 100/3, 100 );
    bin = int(x/5);
    pbins[bin]+=1;

    
  print "tbins = triangle(0,100*2/3)";
  tbins = { bin:0 for bin in range(20) };
  for i in range(0,100000):
    x = triangle( 100*2/3, 0 );
    bin = int(x/5);
    tbins[bin]+=1;
  
  for bin in range(20):
    print "P"*(pbins[bin]/500);
    print "T"*(tbins[bin]/500);
 
 
################################################################################
  
class Tracefile:
  values = [ ("time","8.1f"),
             ("gen","8d"),
             ("pop_size","8d"),
             ("LTETOTAL","8d"),
             ("LTE000pe","8d"),
             ("LTE025pe","8d"),
             ("LTE050pe","8d"),
             ("LTE075pe","8d"),
             ("LTE100pe","8d"),
             ("LTEAUT","8d"),
             ("LTEAUT000pe","8d"),
             ("LTEAUT025pe","8d"),
             ("LTEAUT050pe","8d"),
             ("LTEAUT075pe","8d"),
             ("LTEAUT100pe","8d"),
             ("LTENAUT","8d"),
             ("LTENAUT000pe","8d"),
             ("LTENAUT025pe","8d"),
             ("LTENAUT050pe","8d"),
             ("LTENAUT075pe","8d"),
             ("LTENAUT100pe","8d"),
             ("DTETOTAL","8d"),
             ("DTE000pe","8d"),
             ("DTE025pe","8d"),
             ("DTE050pe","8d"),
             ("DTE075pe","8d"),
             ("DTE100pe","8d"),
             ("DTEAUT","8d"),
             ("DTEAUT000pe","8d"),
             ("DTEAUT025pe","8d"),
             ("DTEAUT050pe","8d"),
             ("DTEAUT075pe","8d"),
             ("DTEAUT100pe","8d"),
             ("DTENAUT","8d"),
             ("DTENAUT000pe","8d"),
             ("DTENAUT025pe","8d"),
             ("DTENAUT050pe","8d"),
             ("DTENAUT075pe","8d"),
             ("DTENAUT100pe","8d"),
             ("FIT000pe","8.6f"),
             ("FIT025pe","8.6f"),
             ("FIT050pe","8.6f"),
             ("FIT075pe","8.6f"),
             ("FIT100pe","8.6f"),
             ("TEDEATH","8d"),
             ("COLLISIO","8d"),
             ("TOTAL_JU","8d"),
             ("LETHAL_J","8d"),
             ("DELETE_J","8d"),
             ("NEUTRA_J","8d"),
             ("BENEFI_J","8d"),
             ("GSIZE000","8d"),
             ("GSIZE025","8d"),
             ("GSIZE050","8d"),
             ("GSIZE075","8d"),
             ("GSIZE100","8d"),
             ("TELOC000","8d"),
             ("TELOC025","8d"),
             ("TELOC050","8d"),
             ("TELOC075","8d"),
             ("TELOC100","8d"),
             ("GELOC000","8d"),
             ("GELOC025","8d"),
             ("GELOC050","8d"),
             ("GELOC075","8d"),
             ("GELOC100","8d"), ];

  headerstr = ", ".join( [ "%8s" % item[0] for item in values ] ) + '\n';
  formatstr = ", ".join( [ "%%(%s)%s" % item for item in values ] ) + '\n';

  def __init__( self, run ):
    file_name = "trace-{:03d}.csv".format(run)
    
    if os.path.exists(file_name):
      self.fp = open(file_name, "a", 1 );	# append
    else:
      self.fp = open(file_name, "w", 1 );	# create
      self.fp.write( self.headerstr );


  def trace( self, valdict ):
    try:
      self.fp.write( self.formatstr % valdict );
    except TypeError:
      print repr( valdict );
      print repr( self.formatstr );
      raise;

  def close( self ):
    self.fp.close();

############################################################################### 

class Experiment:
  def __init__( self, statefile=None ):
    if statefile:
      self.load( statefile );
      output( "LOADING", "Loaded %s" % statefile );
    else:

      test_species1 = Species( 1, [TestChromosome2] );
      self.pop = Population( parameters.Carrying_capacity, test_species1 );

    c0 = self.pop[0].chromosome[0];  # find chromosome 0 in individual
    output( "INITIALIZATION", "Experiment.__init__: pop %d TEs %d genes %d" % \
                ( len(self.pop.individual), len( c0.TEs() ), len( c0.genes() ) ) );

  def save(self, run):
    fp = gzip.open( "state-%03d-%07d.gz" % (run, self.pop.generation_no), "w" );
    fp.write( "random.setstate(%s);\n" % ( repr(random.getstate()), ) );
    fp.write( "self.pop = %s;\n" % (repr(self.pop),) );
    fp.close();

  def load( self, statefile ):
    fp = gzip.open( statefile, "r" );
    exec( fp.read() );
    fp.close();

  def sim_generations( self, run ):
    tf = Tracefile(run);
    self.save(run);	# save state and random state

    tracedict = self.get_tracedict();	# trace entry for initial conditions
    tf.trace( tracedict );

    while self.pop.generation_no < parameters.Maximum_generations:

      output( "GENERATION", "Generation: %d" % self.pop.generation_no );
      te_effects = self.pop.generation();	# run a generation and collect effects

      tracedict = self.get_tracedict();	# trace entry, post generation
      tracedict.update( te_effects );	# add effects
      tf.trace( tracedict );


      if not len( self.pop.individual ) > 0:
        output( "HOST EXTINCTION", "Host extinction after %s generations...sorry :(" % self.pop.generation_no );
        break;

      if not tracedict['LTETOTAL'] > 0:
        output( "TE EXTINCTION", "TE extinction after %s generations...sorry :(" % self.pop.generation_no );
        if hasattr( parameters, "Terminate_no_TEs" ) and parameters.Terminate_no_TEs:
          break;
      if self.pop.generation_no % parameters.save_frequency == 0:
        self.save(run);

    tf.close();

  def get_tracedict( self ):
    # We collect TE data, as well as autonomous/non-autonomous specific data
    live_tes =[ (len(individual.chromosome[0].TEs(live=True, dead=False))) for individual in self.pop.individual];
    live_autonomous_tes =[ (len(individual.chromosome[0].TEs(live=True, dead=False, autonomous=True))) for individual in self.pop.individual];
    live_non_autonomous_tes = [ (len(individual.chromosome[0].TEs(live=True, dead=False, autonomous=False))) for individual in self.pop.individual];

    dead_tes =[ (len(individual.chromosome[0].TEs(live=False, dead=True))) for individual in self.pop.individual];
    dead_autonomous_tes =[ (len(individual.chromosome[0].TEs(live=False, dead=True, autonomous=True))) for individual in self.pop.individual];
    dead_non_autonomous_tes =[ (len(individual.chromosome[0].TEs(live=False, dead=True, autonomous=False))) for individual in self.pop.individual];

    fitnesses = [ individual.fitness for individual in self.pop.individual ];

    tracedict = {
      'time':     time.clock(),
      'gen':      self.pop.generation_no,
      'pop_size': len(self.pop.individual),
      'LTETOTAL': sum(live_tes),
      'LTEAUT': sum(live_autonomous_tes),
      'LTENAUT': sum(live_non_autonomous_tes),
      'DTETOTAL': sum(dead_tes),
      'DTEAUT': sum(dead_autonomous_tes),
      'DTENAUT': sum(dead_non_autonomous_tes),
      'TEDEATH':  0, 	# set effects to zero until we observe them
      'COLLISIO': 0,
      'TOTAL_JU': 0,
      'LETHAL_J': 0,
      'DELETE_J': 0,
      'NEUTRA_J': 0,
      'BENEFI_J': 0,
 };

    tracedict.update( Quartiles('LTE%03dpe', live_tes) );
    tracedict.update( Quartiles('LTEAUT%03dpe', live_autonomous_tes) );
    tracedict.update( Quartiles('LTENAUT%03dpe', live_non_autonomous_tes) );
    tracedict.update( Quartiles('DTE%03dpe', dead_tes) );
    tracedict.update( Quartiles('DTEAUT%03dpe', dead_autonomous_tes) );
    tracedict.update( Quartiles('DTENAUT%03dpe', dead_non_autonomous_tes) );
    tracedict.update( Quartiles('FIT%03dpe', fitnesses) );

    genomesizes = [ individual.chromosome[0].length for individual \
                                in self.pop.individual ];

    tracedict.update( Quartiles('GSIZE%03d',genomesizes) );

    telocs = [ te.start for individual in self.pop.individual \
                        for te in \
                          individual.chromosome[0].TEs(live=True,dead=False) ];

    tracedict.update( Quartiles('TELOC%03d',telocs) );

    gelocs = [ ge.start for individual in self.pop.individual \
                        for ge in individual.chromosome[0].genes() ];

    tracedict.update( Quartiles('GELOC%03d',gelocs) );

    
    return tracedict;

  
   
 
################################################################################

        
################################################################################
# Main code; program starts here
################################################################################

if __name__=="__main__":
  if len( sys.argv ) > 2:
    sys.stderr.write( "You must run it as: python2.7 ../../TEWorldCode/TESim.py, with an optional numerical argument that specifies the number of runs.\n");
    sys.exit(-1);

  num_runs = 1

  if len(sys.argv) == 2:
    num_runs = int(sys.argv[1])
    
  # Iterates by the specified number of runs, or once if unspecified
  for i in range(num_runs):
    run = i + 1
    
    # This ensures that the existing trace files are not overwritten (unless their file name is manually changed)
    run_number = len(glob.glob("trace-???.csv")) + 1
    run_explanation = "The results and state files are stored in files denoted by '-{:03d}' (i.e. trace-{:03d}.csv) files.".format(run_number, run_number)
    print("Run {} started. {}".format(run, run_explanation))
    Experiment( parameters.saved ).sim_generations(run_number)
    print("Run {} completed. {}".format(run, run_explanation))

