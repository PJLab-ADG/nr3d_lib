"""
@file   config.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Utilities for configurations, based on Omegaconf and addict.
"""

import os
import yaml
import addict
import argparse
from numbers import Number
from typing import Sequence
from omegaconf import OmegaConf

# NOTE: eval is evil! This could be dangerous! As it allows arbitary command.
# OmegaConf.register_new_resolver("eval", lambda exp: eval(exp))
# OmegaConf.register_new_resolver("eval", lambda exp: eval_expr(exp))
OmegaConf.register_new_resolver("eval", lambda exp: exp_evaluator.eval(exp))
OmegaConf.register_new_resolver("import", lambda fpath: OmegaConf.load(fpath))

class ConfigDict(addict.Dict):
    # Borrowed from https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/config.py
    def __missing__(self, name):
        raise KeyError(name)
    def __getattr__(self, name):
        try:
            value = super().__getattr__(name)
        except KeyError:
            ex = AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        except Exception as e:
            ex = e
        else:
            return value
        raise ex

def parse_device_ids(device_ids, is_ddp=False):
    if is_ddp:
        print("=> Ignoring device_ids configs when using DDP. Auto set to -1.")
        device_ids = -1
    else:
        if (isinstance(device_ids, Number) and device_ids == -1) \
            or (isinstance(device_ids, list) and len(device_ids) == 0):
            import torch
            device_ids = list(range(torch.cuda.device_count()))
        # # e.g. device_ids: 0 will be parsed as device_ids [0]
        elif isinstance(device_ids, Number):
            device_ids = [device_ids]
        # # e.g. device_ids: 0,1 will be parsed as device_ids [0,1]
        elif isinstance(device_ids, str):
            device_ids = [int(m) for m in device_ids.split(',')]
    print(f"=> Use cuda devices: {device_ids}")
    return device_ids

class BaseConfig:
    def __init__(self, initialize=True):
        if initialize:
            self.parser = self._create_parser()
            self.initialized = True
        else:
            self.initialized = False

    @staticmethod
    def _create_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, default=None, 
                            help="Specifies the path to the config file. You should always specify one of --config or --resume_dir.")
        parser.add_argument('--resume_dir', type=str, default=None, 
                            help="Specifies the directory of the experiment to load/resume. You should always specify one of --config or --resume_dir.")
        parser.add_argument('--device_ids', type=str, default='0', 
                            help="Optionally, specify a target device id. "\
                            "NOTE: Using the environment variable `CUDA_VISIBLE_DEVICES=x` is recommended, "\
                            "as this argument is not always reliable.")
        parser.add_argument('--ddp', action='store_true', help="[DEPRECATED] We discard DDP support for now.")
        parser.add_argument('--port', type=int, default=None, help="[DEPRECATED] We discard DDP support for now.")
        return parser
        
    def parse(
        self, 
        manual_cli_args: Sequence[str] = None,
        base_config_path: str=None,
        parser: argparse.ArgumentParser=None,
        stand_alone=True,
        print_config=True,
        config_path: str=None,
        resume_dir: str=None,
        ) -> ConfigDict:
        """ Parse commond line configs (both argparse and dotlist), yaml config file configs, optional base config file configs, and merge them together into a ConfigDict
            When merging different sources of configs, the priorty orders are:
                Command line dotlist params  --over-->  command line argparse params  --over-->  "--config" config yaml  --over-->  base_config_path config yaml

            Dot-list command-line params are supported for more customized specification. 
                For example, if you have {"training": {"optim": {"lr": 0.01}}} in your config yaml file named "xxx.yaml", 
                    you can specify "--training.optim.lr=0.1" or "training.optim.lr=0.1" after "--config xxx.yaml" to overwrite configs in yaml.
                For more details, check the original documentation of OmegaConf:
                    https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#from-a-dot-list

        Args:
            manual_cli_args (Sequence[str], optional): Manually given command line args to parse (bypass sys.argv). Defaults to None.
            base_config_path (str, optional): Optional basic config file. Defaults to None.
            parser (ArgumentParser, optional): Manually given command line parser (bypass the built-in one). Defaults to None.
            stand_alone (bool, optional): Indicates whether this experiment is a stand-alone or one of multiple experiments in a batch. Defaults to True.
            print_config (bool, optional): Whether print to console the final parsed config. Defaults to True.
            config_path (str, optional): Bypass command line "--config". Defaults to None.
            resume_dir (str, optional): Bypass command line "--resume_dir". Defaults to None.
        
        Returns:
            ConfigDict: The final parsed and merged configs
        """        
        if not self.initialized:
            if parser is None:
                parser = self._create_parser()
        else:
            parser = self.parser
        
        args, unknown = parser.parse_known_args(manual_cli_args)
        
        resume_dir = args.resume_dir if resume_dir is None else resume_dir
        config_path = args.config if config_path is None else config_path
        
        if resume_dir is not None:
            assert config_path is None, "given --config will not be used when given --resume_dir"
            base_config = OmegaConf.create()
            config = OmegaConf.load(os.path.join(resume_dir, 'config.yaml'))

            # Use the loading directory as the experiment path
            config.exp_dir = resume_dir
            print(f"=> Loading previous experiments in: {config.exp_dir}")
        else:
            if base_config_path is not None:
                assert os.path.exists(base_config_path), f"`base_config_path` not exist: {base_config_path}"
                base_config = OmegaConf.load(base_config_path)
            else:
                base_config = OmegaConf.create()

            if config_path is not None:
                assert os.path.exists(config_path), f"`config` not exist: {config_path}"
                config = OmegaConf.load(config_path)
                assert (not stand_alone) or 'exp_dir' in config, 'Please specify exp_dir in task_config'
            elif stand_alone:
                raise RuntimeError("Expect one of --resume_dir or --config")
            else:
                config = OmegaConf.create()

        # Config from argparse
        argparse_config = OmegaConf.create(vars(args))
        argparse_config.pop('config')
        argparse_config.pop('resume_dir')
        
        # Unknown args from command line (will override base config file)
        # Should be in dot-list format: https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#from-a-dot-list
        # Remove leading "--" if present. 
        unknown = [u[2:] if u.startswith('--') else u for u in unknown]
        cli_config = OmegaConf.from_dotlist(unknown)
        
        """ NOTE: Overwrite order
        command line dotlist params  --over-->  command line param  --over-->  args.config  --over-->  default config yaml
        """
        config = OmegaConf.merge(base_config, config, argparse_config, cli_config)       
        
        # NOTE: parse device_ids
        config.device_ids = parse_device_ids(config.get('device_ids', 0), args.ddp)

        if print_config:
            print('')
            print('----------------- Config ---------------\n')
            print(OmegaConf.to_yaml(config, resolve=True))
            print('------------------- End ----------------\n')
            print('')

        """
        Interpolate omegaconf at last
        
        NOTE: (1) Since config interpolation is performed after merging configs, 
            it's possible to use command line args to overwrite the keys that are used to interpolate in config yaml.
            
            For example if you have a config yaml:
                ```yaml
                some_config: 
                  param_a:
                    subparam_1: 20.0
                  param_b:
                    word: hello_world
                some_other_config:
                  param_c: ${some_config.param_a.subparam_1}
                ```
            
            You can change the value of `some_config.param_a.subparam_1` in command line, 
                    (e.g. --some_config.param_a.subparam_1=30.0 )
                which will also affect the value of `some_other_config.param_c`
        
        NOTE: (2) Since OmegaConf's runtime interpolation is of VERY high temporal cost,
            it's important to use the interpolated dict object in training or rendering, rather than the original OmageConf object
            
            For example, codes like `for k,v in cfg.items()` or `if 'foo' in cfg` can take more than 1ms to evaluate, 
                while simple dicts take only take less than a few microseconds.
            
            `for k,v in cfg.items()` benchmark time for 10000 iters:
                omegaconf 0.5495906309224665
                dict 0.0025583491660654545
                addict 0.008531622588634491
                easydict 0.002534216269850731
            
            Hence, we choose `addict` because:
                1. `.` attribute access
                2. set new attr
                3. although slower than dict/easydict, it's acceptable and much faster than Omageconf
        """
        config = ConfigDict(OmegaConf.to_container(config, resolve=True))
        return config

def load_config(path: str):
    cfg = OmegaConf.load(path)
    cfg = ConfigDict(OmegaConf.to_container(cfg, resolve=True)) # Resolve in advance.
    return cfg

def load_config_from_str(cfg: str):
    cfg = OmegaConf.create(cfg)
    cfg = ConfigDict(OmegaConf.to_container(cfg, resolve=True)) # Resolve in advance.
    return cfg

def save_config(cfg: ConfigDict, path: str, ignore_fields=["ddp"]):
    datadict = cfg.copy()
    for field in ignore_fields:
        datadict.pop(field, None)
    with open(path, 'w', encoding='utf8') as outfile:
        if OmegaConf.is_config(datadict):
            outfile.write(OmegaConf.to_yaml(datadict, resolve=True))
        elif isinstance(datadict, ConfigDict):
            # NOTE: Set sort_keys to False to maintain key order, crucial for the sequence of model loading.
            yaml.dump(datadict.to_dict(), outfile, default_flow_style=False, sort_keys=False)
        else:
            raise ValueError(f"Invalid cfg type = {type(datadict)}")

#---------------------------------------------------------
# NOTE: To safely evaluate a string / expression of config 
# Source: https://stackoverflow.com/a/2371789/11121534
# Modified by jianfei guo, 20220117
#---------------------------------------------------------
from pyparsing import (Literal, CaselessLiteral, Word, Keyword, Combine, Group, Optional, ZeroOrMore, Forward, nums, alphas, oneOf)
import math
import operator

class NumericStringParser(object):
    '''
    Most of this code comes from the fourFn.py pyparsing example

    '''
    __author__ = 'Paul McGuire, and Jianfei Guo'
    __version__ = '$Revision: 0.1 $'
    __date__ = '$Date: 2022-01-17 $'
    __source__ = '''http://pyparsing.wikispaces.com/file/view/fourFn.py
    http://pyparsing.wikispaces.com/message/view/home/15549426
    '''
    __note__ = '''
    Rewrap Paul McGuire's fourFn.py as a class, so I can use it
    more easily in other places.
    
    NOTE: modified by jianfei guo: 
    - add '**' operator
    - add logic operators.
    '''

    def pushFirst(self, strg, loc, toks):
        self.exprStack.append(toks[0])

    def pushUMinus(self, strg, loc, toks):
        if toks and toks[0] == '-':
            self.exprStack.append('unary -')

    def pushUNot(self, strg, loc, toks):
        if toks and (toks[0] == 'not' or toks[0] == '~'):
            # unary not
            self.exprStack.append(toks[0])

    def __init__(self):
        """
        expop   :: '^' | '**'
        multop  :: '*' | '/'
        addop   :: '+' | '-'
        integer :: ['+' | '-'] '0'..'9'+
        atom    :: PI | E | real | fn '(' expr ')' | '(' expr ')'
        factor  :: atom [ expop factor ]*
        term    :: factor [ multop factor ]*
        arithm  :: term [ addop term ]*
        NOTE: added by jianfei guo: logic operators.
        andexpr :: arithm [ '&' arithm ]*
        orexpr  :: andexpr [ '|' andexpr ]*
        compexpr:: orexpr [ compop orexpr ]*
        expr    :: compexpr 'not'? compexpr 
        
        """
        point = Literal(".")
        e = CaselessLiteral("E")
        fnumber = Combine(Word("+-" + nums, nums) +
                          Optional(point + Optional(Word(nums))) +
                          Optional(e + Word("+-" + nums, nums)))
        ident = Word(alphas, alphas + nums + "_$")
        plus = Literal("+")
        minus = Literal("-")
        mult = Literal("*")
        div = Literal("/")
        lpar = Literal("(").suppress()
        rpar = Literal(")").suppress()
        addop = plus | minus
        multop = mult | div
        expop = Literal("^") | Literal("**")
        # logic operators, added by jianfei.
        andop = Literal("&")
        orop = Literal("|")
        compop = Literal(">") | Literal("<") | Literal(">=") | Literal("<=") | Literal("!=") | Literal("==") | Keyword("is") | Keyword("isnot")
        notop = Keyword("not") | Literal("~")
        
        pi = CaselessLiteral("PI")
        none = CaselessLiteral("NONE")
        # pi = CaselessKeyword("PI")
        # none = CaselessKeyword("NONE")
        expr = Forward()
        atom = ((Optional(oneOf("- +")) +
                 (ident + lpar + expr + rpar | pi | e | none | fnumber).setParseAction(self.pushFirst))
                | Optional(oneOf("- +")) + Group(lpar + expr + rpar)
                ).setParseAction(self.pushUMinus)
        
        # by defining exponentiation as "atom [ ^ factor ]..." instead of
        # "atom [ ^ atom ]...", we get right-to-left exponents, instead of left-to-right
        # that is, 2^3^2 = 2^(3^2), not (2^3)^2.
        factor = Forward()
        factor << atom + ZeroOrMore((expop + factor).setParseAction(self.pushFirst))
        term = factor + ZeroOrMore((multop + factor).setParseAction(self.pushFirst))
        # expr << term + ZeroOrMore((addop + term).setParseAction(self.pushFirst))
        # NOTE Added by jianfei: logic operators.
        #       priority is according to: https://docs.python.org/3/reference/expressions.html#operator-precedence
        arithm = term + ZeroOrMore((addop + term).setParseAction(self.pushFirst))
        andexpr = arithm + ZeroOrMore((andop + arithm).setParseAction(self.pushFirst))
        orexpr = andexpr + ZeroOrMore((orop + andexpr).setParseAction(self.pushFirst))
        compexpr = orexpr + ZeroOrMore((compop + orexpr).setParseAction(self.pushFirst))
        expr << (Optional(notop) + compexpr).setParseAction(self.pushUNot)
        
        # addop_term = ( addop + term ).setParseAction( self.pushFirst )
        # general_term = term + ZeroOrMore( addop_term ) | OneOrMore( addop_term)
        # expr <<  general_term
        self.bnf = expr
        # map operator symbols to corresponding arithmetic operations
        epsilon = 1e-12
        self.opn = {"+": operator.add,
                    "-": operator.sub,
                    "*": operator.mul,
                    "/": operator.truediv,
                    "^": operator.pow,
                    "**": operator.pow,
                    # logic operators, added by jianfei.
                    "|": operator.or_,
                    "&": operator.and_,
                    ">": operator.gt,
                    "<": operator.lt,
                    ">=": operator.ge,
                    "<=": operator.le,
                    "!=": operator.ne,
                    "==": operator.eq,
                    "is": operator.is_,
                    "isnot": operator.is_not}
        self.fn = {"sin": math.sin, 
                   "cos": math.cos,
                   "tan": math.tan,
                   "exp": math.exp,
                   "sqrt": math.sqrt,
                   "abs": abs,
                   "trunc": lambda a: int(a),
                   "int": lambda a: int(a),
                   "round": round,
                   "floor": math.floor,
                   "sgn": lambda a: abs(a) > epsilon and operator.eq(a, 0) or 0}

    def evaluateStack(self, s):
        op = s.pop()
        if op == 'unary -':
            return -self.evaluateStack(s)
        if op in self.opn.keys():
            op2 = self.evaluateStack(s)
            op1 = self.evaluateStack(s)
            return self.opn[op](op1, op2)
        elif op == "PI":
            return math.pi  # 3.1415926535
        elif op == "E":
            return math.e  # 2.718281828
        elif op == "NONE":  # caseless
            return None
        elif op == "not" or op == "~":
            return not self.evaluateStack(s)
        elif op in self.fn:
            return self.fn[op](self.evaluateStack(s))
        elif op[0].isalpha():
            return 0
        else:
            # return float(op)
            return int(op) if op.isdigit() else float(op)

    def eval(self, num_string, parseAll=True):
        self.exprStack = []
        results = self.bnf.parseString(num_string, parseAll)
        val = self.evaluateStack(self.exprStack[:])
        return val

exp_evaluator = NumericStringParser()


if __name__ == "__main__":
    def test1():
        cfg = \
r"""
model:
  t1: ${eval:"int(2^3)"}
  t2: ${eval:"int(2**3)"}
  t3: ${eval:"int(2**3**2)"}
  t4: ${eval:"sin(2*pi)"}
  t5: ${eval:"e**cos(2*pi)"}
  
  batch_size: 16
  num_epoch: 100
  iter_size: ${eval:"sqrt( ${.batch_size} ) * ${.num_epoch}"}
  iter_size2: ${eval:"sqrt(4)*${.num_epoch}"}
  rayschunk: ${eval:"sqrt(${.batch_size})"}

#   bad: ${eval:"__import__('os')"}

training:
  i_save: ${eval:"100*${model.batch_size}"}
        """
        
        c = OmegaConf.create(cfg)
        print(OmegaConf.to_yaml(c, resolve=True))
        
        # from icecream import ic
        # ic(c.model.t1)
        # ic(c.model.t2)
        # ic(c.model.t3)
        # ic(c.model.t4)
        # ic(c.model.t5)
        # ic(c.model.iter_size)
        # ic(c.model.iter_size2)
        # ic(c.model.rayschunk)
        # ic(c.training.i_save)
        
    def test2():
        cfg = \
r"""
u:
  aa: true
  bb: null
  cc: 1
  t2: ${eval:"pi*3"}
  xx: ${eval:"${.bb} isnot none"}       # null is not None -> False
  yy: ${eval:"${.bb} is none"}          # null is None -> True
  zz: ${eval:"not ${.cc} isnot none"}   # not 1 is not None -> not true -> False
  zz2: ${eval:"not (not ${.cc} isnot none)"}   # not 1 is not None -> not true -> False
  gg: ${eval:"${.cc}>0"}                # 1 > 0 -> True
  hh: ${eval:"${.cc}!=1"}               # 1 != 1 -> True
  ff: ${eval:"3|1+2"}                   # 3|1+2 -> 3|(1+2) -> 3
"""
        c = OmegaConf.create(cfg)
        print(OmegaConf.to_yaml(c, resolve=True))
    
    def test3():
        cfg = \
r"""
d:
  u:
    aa: true
    bb: null
e:
  v:
    ${..d.u}
"""
        c = OmegaConf.create(cfg)
        print(OmegaConf.to_yaml(c, resolve=True))

    def test4():
        cfg = \
r"""
training:
  iter: 3000
id: world
a: hello_${id}
# b: ${eval:"[500, 1000, 1500, 2000]+${training.iter}"}
b:
  - 500
  - 1000
  - 1500
  - 2000
  - ${training.iter}
"""
        c = OmegaConf.create(cfg)
        print(OmegaConf.to_yaml(c, resolve=True))

    # test1()
    # test2()
    # test3()
    test4()