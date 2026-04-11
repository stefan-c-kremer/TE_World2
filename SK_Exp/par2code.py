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
      If the string is not valid Python.
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
# Dictionary of standard parameters and their values that must be present
# in every parameters.py file.
################################################################################

STANDARD_PARAMS = {
  'output': "{'SPLAT': False, 'SPLAT FITNESS': False, 'INITIALIZATION': False, 'GENERATION': True, 'HOST EXTINCTION': True, 'TE EXTINCTION': True, 'TRIAL NO': True, 'GENE INIT': False, 'TE INIT': False}",
  'Gene_length': '1000',
  'Append_gene': 'True',
  'Initial_Aut_TEs': '1',
  'TE_length': 'lambda autonomous: 6000 if autonomous else 300',
  'Host_start_fitness': '1.0',
  'Host_reproduction_rate': '1',
  'Host_survival_rate': 'lambda propfit: min(Carrying_capacity * propfit, 0.95)',
  'TE_excision_rate': '0.0',
  'Maximum_generations': '1500',
  'Terminate_no_TEs': 'True',
  'seed': 'None',
  'save_frequency': '50',
};

################################################################################
# List of parameters which can have a High or Low setting value.
# Some parameters have more than one parameter variables involved.
# (Stefan's version of Isaiah's YAML file.)
################################################################################

HIGH_LOW_PARAMS = [

  # 0->0,1 - Corrected_mutation_rate 
  ( 'Host_mutation_rate', ( '0.3', 
                            '0.03' ) ),

  ( 'Initial_genes', ( '5000', '500' ) ),
  
  # 1->2 - NC_BP
  ( 'Junk_BP', ( '14 * MILLION', 
                 '1.4 * MILLION' ) ),

  # 2->3,4 - Mutation_effect - high mutation effect is higher proliferation?
  ( 'Host_mutation', ( 'ProbabilityTable(0.4, lambda fit: 0.0, 0.3, lambda fit: fit - random.random() * 0.1, 0.15, lambda fit: fit, 0.15, lambda fit: fit + random.random() * 0.1)', 
                       'ProbabilityTable(0.4, lambda fit: 0.0, 0.3, lambda fit: fit - random.random() * 0.01, 0.15, lambda fit: fit, 0.15, lambda fit: fit + random.random() * 0.01)' ) ),

  ( 'Insertion_effect', ( 'ProbabilityTable(0.3, lambda fit: 0.0, 0.2, lambda fit: fit - random.random() * 0.1, 0.3, lambda fit: fit, 0.2, lambda fit: fit + random.random() * 0.1)', 
                          'ProbabilityTable(0.3, lambda fit: 0.0, 0.2, lambda fit: fit - random.random() * 0.01, 0.3, lambda fit: fit, 0.2, lambda fit: fit + random.random() * 0.01)' ) ),


  # 3->5 - Carrying_capacity
  ( 'Carrying_capacity', ( '300', '30' ) ),

  # 4->6 - TE_progeny - 15% change of inserting 3 gives higher proliferation
  ( 'TE_progeny', ( 'ProbabilityTable(0.0, 0, 0.55, 1, 0.3, 2, 0.15, 3)',
                    'ProbabilityTable(0.15, 0, 0.55, 1, 0.3, 2)' ) ),

  # 5->7 TE_death_rate
  ( 'TE_death_rate', ( '0.0005', '0.005' ) ),

  # 6->8,9 - Insertion_bias
  ( 'TE_Insertion_Distribution', ( 'Triangle(pmax=0, pzero=3.0 / 3.0)', 
                                   'Flat()' ) ),

  ( 'Gene_Insertion_Distribution', ( 'Triangle(pzero=1.0 / 3.0, pmax=1)',
                                     'Flat()' ) ),
                                     

  # 7->10 Initial_NAut_TEs
  ( 'Initial_NAut_TEs', ( '3', '1' ) ),

  # 8->11 Kidnapping_frequency
  ( 'Kidnapping_frequency', ( 
            'lambda live_aut, live_naut: 1 - 1 / (1 + 0.07 * live_naut)',
            'lambda live_aut, live_naut: 1 - 1 / (1 + 0.02 * live_naut)',
  ) )
];


################################################################################

def pp_dict( d: dict[ str, str ] ):
  """
  Print the contents of the given dictionary with one key, value pair per
  line.
  """
  print( "{" );
  for key,val in d.items():
    print( f"  {repr(key)}: {repr(val)}," );
  print( "}" );

################################################################################

def check_standard_params( fn:str, d:dict ):
  """
  Check that the dictionary contains all of the standard parameters and that
  their values conform to expectation.
  fn is the filename for exception messages.
  """
  for key,value in STANDARD_PARAMS.items():
    if key not in d:
      raise KeyError( f"{fn}: Missing required STANDARD parameter: '{key}'." );
    if d[key] != STANDARD_PARAMS[key]:
      raise ValueError( f"{fn}: Invalid value for key {key}, expected {repr(STANDARD_PARAMS[key])}, got {repr(d[key])}." );

################################################################################

def pair_check( code: str, pairs: list[tuple[int,int]] ) -> bool:
  """
  Check that parameter pairs have matching H/L values.
  * matches either H or L.
  """
  for pair in pairs:
    if code[pair[0]]=='*' or code[pair[1]]=='*' or code[pair[0]]==code[pair[1]]:
      return True;
    else:
      return False;

################################################################################

def get_code( fn: str, d:dict ) -> str:
  """
  DEPRECATED:  this function no longer works with new encoding.

  Converts the parameters in the dictionary d into a High/Low code string.
  fn is the file name for error reporting.
  """
  code = "";
  for key, values in HIGH_LOW_PARAMS:
    if key not in d:
      d[key] = None;
    if not d[key] in values:
      raise ValueError( f"{fn}: Invalid value for key {key}, expected one of {repr(values)}, got {repr(d[key])}." );
    if d[key] == values[0]:
      if values[0]==values[1]:
        code += "*";    # both values are same
      else:
        code += "H";
    elif d[key] == values[1]:
      code += "L";
    else:
      code += "-";
    

  if not pair_check( code, [ (0,1), (3,4), (9,10), (11,12) ] ):
    code = f"{code} ({code[0:2]}){code[2]}({code[3:5]}){code[5:9]}" + \
           f"({code[9:11]})({code[11:13]}){code[13]}";

    print( d['Host_mutation_rate'] );
    print( HIGH_LOW_PARAMS[0] );
    raise ValueError( f"Inconsistent paired parameter {code}." );
  else:
    code = code[0]+code[2:4]+code[5:10]+code[11]+code[13];    
       # remove duplicate pairs
    
  return code;


################################################################################

def check_dir( directory: str ):
  """
  Print filenames and HL codes for all files in the given directory.
  """

  if directory == "PaperExperiments":
    fmt = 'Old';

  if directory == "TE-Experiments":
    fmt = 'New';

  paramfns = sorted( glob.glob( f"../{directory}/*/parameters.py" ) );



  paramdicts = { fn: pretty_rhs(extract_parameters(fn)) for fn in paramfns };

  for fn, paramdict in paramdicts.items():
    check_standard_params( fn, paramdict );
    print( fn, get_code( fn, paramdict ) );


################################################################################

def code2str( code: str ) -> str: 
  """
  Convert a 8 character of 9 character parameter code to a string
  containing a parameters.py file.
  """

  if len(code)==9 and all( c in {"H","L"} for c in code ) or \
     len(code)==8 and all( c in {"H","L"} for c in code[:7] ) and code[7]=='0':
    pass;
  else:
    raise ValueError( f"Invalid H/L code, '{code}'." );

  # extend code for paired parameters
  code = 2*code[0]+code[1]+2*code[2]+code[3:6]+2*code[6]+code[7]+code[8:];
  if len(code)==11:
    code += 'H';

  d = STANDARD_PARAMS.copy();
  for i,val in enumerate(code):
    if val=='H':
      idx = 0;
    elif val=='L':
      idx = 1;
    elif val=='0':
      d[ HIGH_LOW_PARAMS[i][0] ] = '0';
      continue;
    d[ HIGH_LOW_PARAMS[i][0] ] = HIGH_LOW_PARAMS[i][1][idx];

  str1 = ( "from TEUtil import *;\n"
           "\n"
           "MILLION = 1000000;\n"
           "\n" );

  for key,val in d.items():
    str1 += f"{key} = {(val)}\n";

  return str1;



################################################################################


if __name__ == "__main__":
  #check_dir( "TE-Experiments" );
  print( code2str( "LHHHHHHLL" ) );
