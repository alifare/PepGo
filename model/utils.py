# parse_instance.py
from __future__ import annotations
import inspect, json, types, typing, enum, datetime, decimal, collections.abc as ca
import uuid

MISSING = object()  # 占位：没找到
MAX_DEPTH = 10  # 防止无限递归
BUILTIN_SCALAR = {str, int, float, bool, type(None), bytes, complex}

# parse_class.py
import inspect
import re
from collections import OrderedDict
from typing import Any, Dict, List, Tuple, Optional

import inspect
import numpy as np
import torch
import pprint as pp


class UTILS:
    def __init__(self):
        super().__init__()

    def parse_var(self, var, label=None):
        callers_local_vars = inspect.currentframe().f_back.f_locals.items()
        var_name = [var_name for var_name, var_val in callers_local_vars if var_val is var]

        if(var_name):
            if(label):
                var_name = '_'.join(var_name+[label])
            else:
                var_name = var_name[0]
        else:
            var_name=label
        
        print('-'*100)

        print(var_name,end=':')
        if(torch.is_tensor(var)):
            print(var.shape,end=' ')
            print(var.dtype,end=' ')
        elif(isinstance(var, list)):
            print(len(var),end=' ')
        elif np.isscalar(var):
            print(var)
        print(type(var))

        pp.pprint(var)



    # ------------- 小工具 -------------
    def _signature(self, f) -> str:
        """返回人类可读的签名，如 'foo(a, b=1, *args, **kwargs)'"""
        sig = inspect.signature(f)
        return str(sig).replace(' -> None', '')

    def _source(self, obj) -> Optional[str]:
        """返回源码，如果拿不到就 None"""
        try:
            return inspect.getsource(obj)
        except (OSError, TypeError):
            return None

    def _doc(self, obj) -> Optional[str]:
        """返回清洗后的 docstring"""
        return inspect.getdoc(obj)

    def parse_class(self, cls):
        self.pprint(self.parse(cls))
        print('-'*100)

    # ------------- 核心逻辑 -------------
    def parse(self, cls: type) -> OrderedDict:
        """解析一个类，返回有序字典"""
        if not inspect.isclass(cls):
            raise TypeError(f'{cls!r} 不是一个类')

        info = OrderedDict()
        info['name'] = cls.__name__
        info['module'] = cls.__module__
        info['mro'] = [c.__name__ for c in cls.__mro__]
        info['doc'] = self._doc(cls)

        # 1. 类属性（定义在类体里、不在方法里的变量）
        info['class_attrs'] = OrderedDict()
        for name, value in cls.__dict__.items():
            if not callable(value) and not name.startswith('_'):
                info['class_attrs'][name] = repr(value)

        # 2. 方法
        methods: Dict[str, Dict[str, Any]] = OrderedDict()
        for name, method in inspect.getmembers(cls, predicate=inspect.isroutine):
            if name.startswith('_') and name != '__init__':
                continue
            methods[name] = OrderedDict()
            methods[name]['signature'] = self._signature(method)
            methods[name]['doc'] = self._doc(method)
            methods[name]['source'] = self._source(method)
            # 区分方法种类
            if isinstance(inspect.getattr_static(cls, name), staticmethod):
                methods[name]['type'] = 'staticmethod'
            elif isinstance(inspect.getattr_static(cls, name), classmethod):
                methods[name]['type'] = 'classmethod'
            elif isinstance(inspect.getattr_static(cls, name), property):
                methods[name]['type'] = 'property'
            else:
                methods[name]['type'] = 'method'
        info['methods'] = methods

        # 3. 构造器单独拿出来
        if '__init__' in methods:
            info['constructor'] = methods.pop('__init__')

        # 4. 全部源码
        info['source'] = self._source(cls)
        return info

    # ------------- 漂亮的打印 -------------
    def pprint(self, info: OrderedDict, width: int = 88):
        """把 parse 的结果打印成人类可读报告"""
        sep = lambda: print('-' * width)
        sep()
        print(f"Class : {info['name']}  (from {info['module']})")
        print(f"MRO   : {' -> '.join(info['mro'])}")
        if info['doc']:
            print("Doc   :\n" + inspect.cleandoc(info['doc']))
        sep()

        if info['class_attrs']:
            print("Class attributes:")
            for k, v in info['class_attrs'].items():
                print(f"  {k} = {v}")
            sep()

        if 'constructor' in info:
            c = info['constructor']
            print(f"__init__{c['signature']}")
            if c['doc']:
                print('  ' + c['doc'].replace('\n', '\n  '))
            sep()

        if info['methods']:
            print("Methods:")
            for name, meta in info['methods'].items():
                print(f"{meta['type']:>12} : {name}{meta['signature']}")
                if meta['doc']:
                    print('  ' + meta['doc'].split('\n')[0])
            sep()

        if info['source']:
            print("Full source:")
            print(info['source'])
        else:
            print("Source not available (maybe from extension or REPL).")


    def _is_scalar(self, obj):
        return type(obj) in BUILTIN_SCALAR or isinstance(obj, (decimal.Decimal, datetime.date, uuid.UUID))

    def _serialize(self, obj, depth: int = 0, seen: dict[int, str] | None = None) -> typing.Any:
        """把任意 Python 对象变成纯 JSON 可序列化结构"""
        if seen is None:
            seen = {}
        obj_id = id(obj)
        if obj_id in seen:  # 循环引用
            return f"<CircularRef to {seen[obj_id]}>"
        if depth > MAX_DEPTH:
            return "<MaxDepth>"

        # 1. 标量直接返回
        if self._is_scalar(obj):
            return obj

        # 2. 枚举
        if isinstance(obj, enum.Enum):
            return obj.name

        # 3. bytes → base64
        if isinstance(obj, bytes):
            return obj.decode('latin1')

        # 4. 函数、类、模块、文件等不可序列化 → repr
        if isinstance(obj, (types.FunctionType, types.MethodType, type, types.ModuleType)):
            return f"<{obj!r}>"

        # 5. 容器：dict / list / tuple / set
        if isinstance(obj, dict):
            seen[obj_id] = f"dict(len={len(obj)})"
            return {str(k): self._serialize(v, depth + 1, seen) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set, frozenset)):
            name = type(obj).__name__
            seen[obj_id] = f"{name}(len={len(obj)})"
            return [self._serialize(i, depth + 1, seen) for i in obj]

        # 6. namedtuple
        if isinstance(obj, tuple) and hasattr(obj, '_fields'):
            seen[obj_id] = f"{type(obj).__name__}"
            return {f: self._serialize(getattr(obj, f), depth + 1, seen) for f in obj._fields}

        # 7. dataclass
        if hasattr(obj, '__dataclass_fields__'):
            seen[obj_id] = f"{type(obj).__name__}"
            return {f: self._serialize(getattr(obj, f), depth + 1, seen)
                    for f in obj.__dataclass_fields__}

        # 8. 通用对象：先抓 __dict__，再抓 __slots__
        cls = type(obj)
        name = f"{cls.__module__}.{cls.__qualname__}"
        seen[obj_id] = name

        data = OrderedDict()
        # 8.1 __dict__
        if hasattr(obj, '__dict__'):
            data.update({k: self._serialize(v, depth + 1, seen) for k, v in vars(obj).items()})

        # 8.2 __slots__
        slots = getattr(cls, '__slots__', ())
        if slots and isinstance(slots, str):
            slots = (slots,)
        for slot in slots:
            val = getattr(obj, slot, MISSING)
            if val is not MISSING:
                data[slot] = self._serialize(val, depth + 1, seen)

        # 8.3 property 等描述符（可选）
        for k, v in inspect.getmembers(cls):
            if k.startswith('_'):
                continue
            if isinstance(v, property) and v.fget:
                try:
                    data[f'<property {k}>'] = self._serialize(getattr(obj, k), depth + 1, seen)
                except Exception as e:
                    data[f'<property {k}>'] = f"<Error: {e}>"

        data['<type>'] = name
        return data

    # ------------------------------------------------
    # 用户只需要这一行
    def parse_instance(self, obj: typing.Any, *, to_json: bool = True) -> typing.Any:
        tree = self._serialize(obj)
        dump_info = json.dumps(tree, ensure_ascii=False, indent=2) if to_json else tree
        pp.pprint(dump_info)
        print('-'*100)
        return(dump_info)