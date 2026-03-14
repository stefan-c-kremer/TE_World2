import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# --- Mapping Dictionaries ---
csv2key = {
    'pop_size': 'hosts', 'LTETOTAL': 'TEs', 'LTEAUT': 'Autonomous TEs', 
    'LTENAUT': 'Non-autonomous TEs', 'LETHAL_J': 'Lethal', 'DELETE_J': 'Delet.', 
    'NEUTRA_J': 'Neutral', 'BENEFI_J': 'Benfit.', 'TEDEATH': 'TEs', 
    'COLLISIO': 'TEs', 'TOTAL_JU': 'TEs'
}

csv2lt = {
    'pop_size': '#FF0000', 'LTETOTAL': '#FF0000', 'LTEAUT': "#00FF26", 
    'LTENAUT': "#C300FF", 'DELETE_J': '#00FF00', 'NEUTRA_J': '#0000FF', 
    'BENEFI_J': '#000000', 'LETHAL_J': '#FF0000'
}

plots_config = [
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
    ( 'TE and Gene Locations', 'gen', ['GSIZE100','GSIZE075','GSIZE050','GSIZE025','GSIZE000', 'GELOC100','GELOC075','GELOC050','GELOC025','GELOC000', 'TELOC100','TELOC075','TELOC050','TELOC025','TELOC000' ] ),
    ( 'Live (autonomous and non-autonomous) TEs vs Generation', 'gen', ['LTEAUT', 'LTENAUT']) 
];
    

if __name__=="__main__":
  if len( sys.argv )!=1:
    sys.stderr.write( "Usage:  python3 ../../TEWorldCode/MakeGraphs.py\n");
    sys.exit(-1);
  
  files = [ int(name[6:9]) for name in glob.glob("trace-???.csv") ];
  files.sort();
  
  for file in files:
    trial = Trial(file);
    for plot in plots:
      try:
        trial.plot( *plot );
      except ValueError as e:   # accomodate old csv files with less columns
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
