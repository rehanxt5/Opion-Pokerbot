#!/usr/bin/env python3
"""Build opion_submission_final/bot.py: the opion.py submission with every
comment and docstring removed (AST strip + unparse), no shebang, no
identifying strings, weights re-injected from the current submissions/opion.py
blobs. Then zip bot.py at the top level and verify all size limits."""
import ast
import os
import re
import zipfile

ROOT = "/Users/luciferxt5/Documents/Coding/pokerbot"
TPL = os.path.join(ROOT, "submissions/opion_template.py")
CUR = os.path.join(ROOT, "submissions/opion.py")
OUTD = os.path.join(ROOT, "submissions/opion_submission_final")
BOT = os.path.join(OUTD, "bot.py")
ZIP = os.path.join(OUTD, "submission.zip")

BANNED = ["lucifer", "opion", "Opion", "OPION", "synergy", "Synergy", "SYNERGY",
          "/Users", "elitev6", "policy_np", "policy_rl", "PPO", "distill",
          "teacher", "colab", "e2b", "E2B", "author", "Author", "gmail",
          "final_disc", "np_policy", "ckpts"]


class DocStrip(ast.NodeTransformer):
    def _strip(self, node):
        b = node.body
        if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) \
                and isinstance(b[0].value.value, str):
            node.body = [ast.Pass()] if len(b) == 1 else b[1:]
        return node

    def visit_Module(self, node):
        self.generic_visit(node)
        return self._strip(node)

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        return self._strip(node)

    def visit_AsyncFunctionDef(self, node):
        self.generic_visit(node)
        return self._strip(node)

    def visit_ClassDef(self, node):
        self.generic_visit(node)
        return self._strip(node)


def main():
    os.makedirs(OUTD, exist_ok=True)
    cur = open(CUR).read()
    act = re.search(r'_ACT_NPZ_B64 = """(.*?)"""', cur, re.S).group(1)
    disc = re.search(r'_DISC_NPZ_B64 = """(.*?)"""', cur, re.S).group(1)
    assert "\n" not in act and "\n" not in disc
    assert "#" not in act and "#" not in disc

    tree = DocStrip().visit(ast.parse(open(TPL).read()))
    ast.fix_missing_locations(tree)
    code = ast.unparse(tree)
    assert "'__ACT_B64__'" in code and "'__DISC_B64__'" in code
    code = code.replace("'__ACT_B64__'", repr(act)).replace("'__DISC_B64__'", repr(disc))

    compile(code, "bot.py", "exec")
    assert not code.startswith("#!")
    assert "#" not in code, "comment character survived the strip"
    assert '"""' not in code, "docstring survived the strip"
    scan = re.sub(r"'[A-Za-z0-9+/=]{1000,}'", "'<BLOB>'", code)
    hits = [b for b in BANNED if b in scan]
    assert not hits, f"banned identifiers present: {hits}"

    open(BOT, "w").write(code)
    print(f"bot.py: {os.path.getsize(BOT):,} B "
          f"({os.path.getsize(BOT) / 1e6:.3f} MB extracted, limit 5 MB)")

    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(BOT, "bot.py")
    with zipfile.ZipFile(ZIP) as z:
        assert z.testzip() is None
        names = z.namelist()
        infos = z.infolist()
    assert names == ["bot.py"], names
    print(f"submission.zip: {os.path.getsize(ZIP):,} B "
          f"({os.path.getsize(ZIP) / 1e6:.3f} MB zipped, limit 5 MB); "
          f"contents {names}; extracts to "
          f"{sum(i.file_size for i in infos):,} B")

    pre = ast.unparse(DocStrip().visit(ast.parse(open(TPL).read())))
    pre = pre.replace("'__ACT_B64__'", repr(act)).replace("'__DISC_B64__'", repr(disc))
    same = ast.dump(ast.parse(pre)) == ast.dump(ast.parse(code))
    print(f"AST equivalence (template-minus-docstrings == bot.py): {same}")
    assert same
    print("BUILD_OK")


if __name__ == "__main__":
    main()
