import re;
import sys;
import ast;
import glob;
from pathlib import Path;

################################################################################

def extract_parameters( path: str ) -> dict[ str, ast.AST ]:
  """
  Parse a Python parameter file and return a mapping from parameter
  names to the AST node of the RHS expression.

  Supports simple assignments and chained assignments at top level.

  Example:
    a = 1
    b = c = Flat()

  becomes:
    {
      "a": Constant( 1 ),
      "b": Call( ... ),
      "c": Call( ... ),
    }

  Raises:
    SyntaxError:
      If the file is not valid Python.
    ValueError:
      If a top-level assignment target is not a simple name.
  """

  text = Path( path ).read_text( encoding="utf-8" );
  tree = ast.parse( text, filename=path );

  params = { };

  for stmt in tree.body:
    if isinstance( stmt, ast.Assign ):
      rhs = stmt.value;

      for target in stmt.targets:
        if not isinstance( target, ast.Name ):
          raise ValueError(
            f"{path}: unsupported assignment target "
            f"{ast.dump( target, include_attributes=False )}"
          );
        params[ target.id ] = rhs;

    elif isinstance( stmt, ast.AnnAssign ):
      if not isinstance( stmt.target, ast.Name ):
        raise ValueError(
          f"{path}: unsupported annotated target "
          f"{ast.dump( stmt.target, include_attributes=False )}"
        );
      if stmt.value is None:
        raise ValueError(
          f"{path}: annotation without value for {stmt.target.id}"
        );
      params[ stmt.target.id ] = stmt.value;

    elif isinstance( stmt, ast.Expr ):
      # Allow harmless top-level docstrings if present.
      if not (
        isinstance( stmt.value, ast.Constant )
        and isinstance( stmt.value.value, str )
      ):
        raise ValueError(
          f"{path}: unsupported top-level expression "
          f"{ast.dump( stmt, include_attributes=False )}"
        );
    elif isinstance( stmt, ( ast.Import, ast.ImportFrom ) ):
      continue;
    else:
      raise ValueError(
        f"{path}: unsupported top-level statement "
        f"{type( stmt ).__name__}"
      );

  return params;

################################################################################

def pretty_rhs( params: dict[ str, ast.AST ] ) -> dict[ str, str ]:
  """
  Return a readable Python source representation of an RHS AST.
  """
  return { key: ast.unparse( node ) for key,node in params.items() };


################################################################################

STANDARD_PARAMS = {
  'output': "{'SPLAT': False, 'SPLAT FITNESS': False, 'INITIALIZATION': False, 'GENERATION': True, 'HOST EXTINCTION': True, 'TE EXTINCTION': True, 'TRIAL NO': True, 'GENE INIT': False, 'TE INIT': False}",
  'Gene_length': '1000',
  'TE_length': '1000',
  'Append_gene': 'True',
  'Initial_TEs': '1',
  'MILLION': '1000000',
  'Host_start_fitness': '1.0',
  'Host_reproduction_rate': '1',
  'Host_survival_rate': 'lambda propfit: min(Carrying_capacity * propfit, 0.95)',
  'Maximum_generations': '1500',
  'Terminate_no_TEs': 'True',
  'seed': 'None',
  'save_frequency': '50',
};

################################################################################

HIGH_LOW_PARAMS = [

  # 0 - TE_progeny - 15% change of inserting 3 gives higher proliferation
  ( 'TE_progeny', ( 'ProbabilityTable(0.0, 0, 0.55, 1, 0.3, 2, 0.15, 3)',
                    'ProbabilityTable(0.15, 0, 0.55, 1, 0.3, 2)' ) ),

  # 1 - TE_excision_rate - higher values gives higher proliferation
  ( 'TE_excision_rate', ( '0.5', '0.1' ) ),

  # 2 - TE_death_rate - lower value gives higher proliferation
  ( 'TE_death_rate', ('0.0005', '0.005') ),

  # 3,4 - Insertion_bias
  ( 'TE_Insertion_Distribution', ( 'Triangle(pmax=0, pzero=3.0 / 3.0)', 
                                   'Flat()' ) ),

  ( 'Gene_Insertion_Distribution', ( 'Triangle(pzero=1.0 / 3.0, pmax=1)',
                                     'Flat()' ) ),
                                     
  # 5,6 - Corrected_mutation_rate 
  ( 'Host_mutation_rate', ( '0.3', 
                            '0.03' ) ),

  ( 'Initial_genes', ( '5000', '500' ) ),

  # 7 - NC_BP
  ( 'Junk_BP', ( '14 * MILLION', 
                 '1.4 * MILLION' ) ),

  # 8,9 - Mutation_effect - high mutation effect is higher proliferation?
  ( 'Host_mutation', ( 'ProbabilityTable(0.4, lambda fit: 0.0, 0.3, lambda fit: fit - random.random() * 0.1, 0.15, lambda fit: fit, 0.15, lambda fit: fit + random.random() * 0.1)', 
                       'ProbabilityTable(0.4, lambda fit: 0.0, 0.3, lambda fit: fit - random.random() * 0.01, 0.15, lambda fit: fit, 0.15, lambda fit: fit + random.random() * 0.01)' ) ),

  ( 'Insertion_effect', ( 'ProbabilityTable(0.3, lambda fit: 0.0, 0.2, lambda fit: fit - random.random() * 0.1, 0.3, lambda fit: fit, 0.2, lambda fit: fit + random.random() * 0.1)', 
                          'ProbabilityTable(0.3, lambda fit: 0.0, 0.2, lambda fit: fit - random.random() * 0.01, 0.3, lambda fit: fit, 0.2, lambda fit: fit + random.random() * 0.01)' ) ),


  # 10 - Carrying_capacity
  ( 'Carrying_capacity', ( '300', '30' ) ),
];

################################################################################

def pp_dict( d: dict[ str, str ] ):
  print( "{" );
  for key,val in d.items():
    print( f"  {repr(key)}: {repr(val)}," );
  print( "}" );

################################################################################

def check_standard_params( fn:str, d:dict ):
  for key,value in STANDARD_PARAMS.items():
    if key not in d:
      raise KeyError( f"{fn}: Missing required STANDARD parameter: '{key}'." );
    if d[key] != STANDARD_PARAMS[key]:
      raise ValueError( f"{fn}: Invalid value for key {key}, expected {repr(STANDARD_PARAMS[key])}, got {repr(d[key])}." );

################################################################################

def get_code( fn: str, d:dict ):
  code = "";
  for key, values in HIGH_LOW_PARAMS:
    if key not in d:
      raise KeyError( f"{fn}: Missing required HIGH_LOW parameter: '{key}'." );
    if not d[key] in values:
      raise ValueError( f"{fn}: Invalid value for key {key}, expected one of {repr(values)}, got {repr(d[key])}." );
    if d[key] in values[0]:
      code += "H";
    else:
      code += "L";

  code = code + " " + code[0:3]+"("+code[3:5]+")"+"("+code[5:7]+")"+code[7]+"("+code[8:10]+")"+code[10];
  return code;



################################################################################

def main( argv: list[ str ] ): 

  #ex="Stefan";
  ex="Isaiah";

  if ex=="Stefan": 
    paramfns = sorted( glob.glob( "../PaperExperiments/*/parameters.py" ) );
  else:
    # fix Isaiahisms
    del( STANDARD_PARAMS['MILLION'] );
    HIGH_LOW_PARAMS[7] =   ( 'Junk_BP', ( '14000000',
                                          '1400000' ) );
    paramfns = sorted( glob.glob( "../TE-Experiments/*/parameters.py" ) );


  paramdicts = { fn: pretty_rhs(extract_parameters(fn)) for fn in paramfns };

  for fn, paramdict in paramdicts.items():
    check_standard_params( fn, paramdict );
    print( fn, get_code( fn, paramdict ) );



  keys = next(iter(paramdicts.values())); # get keys from first param dictionary

  summary = {
    key: set( paramdict[key] for paramdict in paramdicts.values() )
           for key in keys

  };

  #pp_dict( summary );
  fixed = {};
  hl = {};
  other = {};
  for key,values in summary.items():
    if len( values )==1:
      fixed[ key ] = next(iter(values));
    elif len( values )==2:
      hl[ key ] = values;
    else:
      other[ key ] = values;


  #pp_dict( fixed );

  #for key,val in hl.items():
    #print( key, val );

  #for key,val in other.items():
    #print( key, val );


if __name__ == "__main__":
  main( sys.argv );
