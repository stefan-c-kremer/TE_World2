import os;
import sys;
import glob;
import subprocess;

 
################################################################################

gnuplot_commands = """
set size ratio 0.2
set title "%s"
set key left
set terminal svg size 800,200 fname 'Verdana' fsize 10
set output '%s-%03d.svg'
plot 'trace-%03d.csv' """;

class Graph:
  def __init__( self, trial, title, x, y ):
    self.trial = trial;
    self.title = title;
    self.x = x;
    self.y = y;

  def plot( self ):
    gnuplot = subprocess.Popen( "gnuplot", stdin=subprocess.PIPE );
    xi = self.trial.headers.index(self.x);
    commands = gnuplot_commands + \
	", '' ".join( 'using %d:%d lt rgb "%s" with lines title "%s"' % (xi, self.trial.headers.index(y1),csv2lt[y1],csv2key[y1]) \
             for y1 in self.y ) + "\nquit\n";

    commands2 = commands % tuple( 2*[self.title] + 2*[self.trial.i] );
    print repr( commands2 );
    gnuplot.communicate( input = commands2 );
    
 

class Trial: 
  def __init__(self,i):
    self.i = i;
    fp = open( 'trace-%03d.csv' % i );
    self.headers = ['#'] + [h.strip() for h in fp.readline().split(',')];
    fp.close();

  def plot( self, title, x, y ):
    Graph( self, title, x, y ).plot();


        
################################################################################
# Main code; program starts here
################################################################################

csv2key = {
  'time':     'time (s)', 
  'gen':      'gens', 
  'pop_size': 'hosts', 
  'LTETOTAL': 'TEs', 
  'LTE000pe': '0%%', 
  'LTE025pe': '25%%', 
  'LTE050pe': '50%%', 
  'LTE075pe': '75%%', 
  'LTE100pe': '100%%', 
  'DTETOTAL': 'TEs', 
  'DTE000pe': '0%%', 
  'DTE025pe': '25%%', 
  'DTE050pe': '50%%', 
  'DTE075pe': '75%%', 
  'DTE100pe': '100%%', 
  'FIT000pe': '0%%', 
  'FIT025pe': '25%%', 
  'FIT050pe': '50%%', 
  'FIT075pe': '75%%', 
  'FIT100pe': '100%%', 
  'TEDEATH':  'TEs', 
  'COLLISIO': 'TEs', 
  'TOTAL_JU': 'TEs', 
  'LETHAL_J': 'Lethal', 
  'DELETE_J': 'Delet.', 
  'NEUTRA_J': 'Neutral', 
  'BENEFI_J': 'Benfit.',
  'GSIZE000': '0%%',
  'GSIZE025': '25%%', 
  'GSIZE050': '50%%',
  'GSIZE075': '75%%',
  'GSIZE100': '100%%',
  'TELOC000': 'T0%%',
  'TELOC025': 'T25%%', 
  'TELOC050': 'T50%%', 
  'TELOC075': 'T75%%', 
  'TELOC100': 'T100%%', 
  'GELOC000': 'G0%%', 
  'GELOC025': 'G25%%', 
  'GELOC050': 'G50%%', 
  'GELOC075': 'G75%%', 
  'GELOC100': 'G100%%'
};

csv2lt = {
  'time':     '#FF0000',
  'gen':      '#FF0000',
  'pop_size': '#FF0000',
  'LTETOTAL': '#FF0000',
  'LTE000pe': '#FFAFAF',
  'LTE025pe': '#FF7F7F',
  'LTE050pe': '#FF0000',
  'LTE075pe': '#FF7F7F',
  'LTE100pe': '#FFAFAF',
  'DTETOTAL': '#FF0000',
  'DTE000pe': '#FFAFAF',
  'DTE025pe': '#FF7F7F',
  'DTE050pe': '#FF0000',
  'DTE075pe': '#FF7F7F',
  'DTE100pe': '#FFAFAF',
  'FIT000pe': '#FFAFAF',
  'FIT025pe': '#FF7F7F',
  'FIT050pe': '#FF0000',
  'FIT075pe': '#FF7F7F',
  'FIT100pe': '#FFAFAF',
  'TEDEATH':  '#FF0000',
  'COLLISIO': '#FF0000',
  'TOTAL_JU': '#FF0000',
  'LETHAL_J': '#FF0000',
  'DELETE_J': '#00FF00',
  'NEUTRA_J': '#0000FF',
  'BENEFI_J': '#000000',
  'GSIZE000': '#FFAFAF',
  'GSIZE025': '#FF7F7F',
  'GSIZE050': '#FF0000',
  'GSIZE075': '#FF7F7F',
  'GSIZE100': '#FFAFAF',
  'TELOC000': '#AFFFAF',
  'TELOC025': '#7FFF7F',
  'TELOC050': '#00FF00',
  'TELOC075': '#7FFF7F',
  'TELOC100': '#AFFFAF',
  'GELOC000': '#AFAFFF',
  'GELOC025': '#7F7FFF',
  'GELOC050': '#0000FF',
  'GELOC075': '#7F7FFF',
  'GELOC100': '#AFAFFF' 
}

plots = [
    ( 'Host Population vs Generation', 'gen', ['pop_size'] ),
    ( 'Total Live TEs vs Generation', 'gen', ['LTETOTAL'] ),
    ( 'Live TE Percentiles vs Generation', 'gen', [ 'LTE100pe',
                           'LTE075pe', 'LTE050pe', 'LTE025pe', 'LTE000pe' ] ),
    ( 'Total Dead TEs vs Generation', 'gen', ['DTETOTAL'] ),
    ( 'Dead TE Percentiles vs Generation', 'gen', [ 'DTE100pe',
                           'DTE075pe', 'DTE050pe', 'DTE025pe', 'DTE000pe' ] ),
    ( 'Fitness Percentiles vs Generation', 'gen', [ 'FIT100pe',
                           'FIT075pe', 'FIT050pe', 'FIT025pe', 'FIT000pe' ] ),
    ( 'TE Deaths vs Generation', 'gen', [ 'TEDEATH' ] ),
    ( 'TE Collisions vs Generation', 'gen', [ 'COLLISIO' ] ),
    ( 'TE Jumps vs Generation', 'gen', [ 'TOTAL_JU' ] ),
    ( 'TE Jump Effects vs Generation', 'gen', ['LETHAL_J', 'DELETE_J', 'NEUTRA_J', 'BENEFI_J' ] ),
    ( 'TE and Gene Locations', 'gen', ['GSIZE100','GSIZE075','GSIZE050','GSIZE025','GSIZE000', 'GELOC100','GELOC075','GELOC050','GELOC025','GELOC000', 'TELOC100','TELOC075','TELOC050','TELOC025','TELOC000' ] ) 
];
    

if __name__=="__main__":
  if len( sys.argv )!=1:
    sys.stderr.write( "Usage:  python2.7 ../../TEWorldCode/MakeGraphs.py\n");
    sys.exit(-1);
  
  files = [ int(name[6:9]) for name in glob.glob("trace-???.csv") ];
  files.sort();
  
  for file in files:
    trial = Trial(file);
    for plot in plots:
      try:
        trial.plot( *plot );
      except ValueError, e:   # accomodate old csv files with less columns
        pass;


    fp = open( "graphs-%03d.html" % file, "w" );
    nums = tuple(6*[file]);
    fp.write( """
<html>
<body>
  <h1> trace-%03d </h1>
""" % file );

    for plot in plots:
      filename = "%s-%03d.svg" % (plot[0],file);
      if os.path.exists(filename):
        fp.write( """
          <p><img src="%s"/></p>
          """ % (filename,) );

    fp.write( """
</body>
</html>
""");

    fp.close();
