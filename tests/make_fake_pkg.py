#!/usr/bin/env python3
"""Makes a small fake PKG for pkg_extract_test, and the folder lpn-pkg should extract from it.

    make_fake_pkg.py <src/pkg/keys.h> <out.pkg> <expected folder> [game|update|dlc]

The PKG is laid out as fpkg tools make them: entry keys and an image key sealed with the fake
keys, and an encrypted, signed outer PFS image whose uroot holds a compressed (PFSC) inner
image, pfs_image.dat, with the game's files. Needs the cryptography package.
"""
import os
import re
import struct
import sys
import zlib
import hashlib
import hmac

from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

keys_h, out_pkg, expected = sys.argv[1:4]
kind = sys.argv[4] if len(sys.argv) > 4 else "game"
src = open(keys_h).read()


def key(name):
    body = re.search(r"k" + name + r"Modulus\[256\] = \{([^}]*)\}", src).group(1)
    n = int.from_bytes(bytes(int(x, 16) for x in re.findall(r"0x([0-9A-F]{2})", body)), "big")
    return rsa.RSAPublicNumbers(65537, n).public_key()


fake, dk3key = key("FakeKeyset"), key("PkgDerivedKey3Keyset")
rnd = __import__("random").Random(7)


def rbytes(n):
    return bytes(rnd.getrandbits(8) for _ in range(n))


# The game's files.
files = {
    "eboot.bin": b"\x7fELF fake eboot " * 300,
    "sce_module/libc.prx": bytes(range(256)) * 900 + rbytes(5000),  # several blocks
    "sce_module/empty.prx": b"",
    "data/noise.bin": rbytes(70000),  # does not compress: stored blocks
    "data/deep/x.txt": b"hello\n",
}
if kind == "update":  # what the update changes, and a file it adds
    files = {"eboot.bin": b"\x7fELF patched eboot " * 200, "data/patch/new.bin": rbytes(3000)}
elif kind == "dlc":
    files = {"costume/kenshiro.dat": rbytes(9000)}


def pfs_image(blocksize, signed, files_tree, compressed_inos=()):
    """A PFS image: [header][inode blocks][dir and file blocks]. Inode 0 super root, 1 uroot."""
    isz = 0x2C8 if signed else 0xA8
    nodes = []  # (kind, name path, data or children)
    dirs = {"": []}
    for path in sorted(files_tree):
        parts = path.split("/")
        for i in range(1, len(parts)):
            d = "/".join(parts[:i])
            if d not in dirs:
                dirs[d] = []
                dirs["/".join(parts[: i - 1])].append(("d", parts[i - 1], d))
        dirs["/".join(parts[:-1])].append(("f", parts[-1], path))
    # inode numbers
    ino = {"<super>": 0, "": 1}
    order = ["<super>", ""]
    for d in sorted(k for k in dirs if k):
        ino[d] = len(order)
        order.append(d)
    for f in sorted(files_tree):
        ino["f:" + f] = len(order)
        order.append("f:" + f)
    count = len(order)
    per = blocksize // isz
    iblocks = (count + per - 1) // per
    block = 1 + iblocks
    content = {}  # inode -> (block, nblocks, data)

    def dirent(i, t, name):
        name = name.encode()
        size = (17 + len(name) + 7) // 8 * 8
        return struct.pack("<IiiI", i, t, len(name), size) + name + b"\0" * (size - 16 - len(name))

    def dir_data(key, parent, kids):
        d = dirent(ino[key], 4, ".") + dirent(ino[parent], 5, "..")
        for kind, name, k in kids:
            d += dirent(ino[k] if kind == "d" else ino["f:" + k], 3 if kind == "d" else 2, name)
        assert len(d) <= blocksize
        return d

    blobs = {0: dir_data("<super>", "<super>", [("d", "uroot", "")])}
    blobs[0] = dirent(0, 4, ".") + dirent(0, 5, "..") + dirent(1, 3, "uroot")
    for d, kids in dirs.items():
        blobs[ino[d]] = dir_data(d, "/".join(d.split("/")[:-1]) if d else "<super>", kids)
    for f, data in files_tree.items():
        blobs[ino["f:" + f]] = data
    for i in range(count):
        data = blobs[i]
        n = max(1, (len(data) + blocksize - 1) // blocksize)
        content[i] = (block, n, data)
        block += n
    total = block
    img = bytearray(total * blocksize)
    mode = (1 if signed else 0) | 8
    hdr = struct.pack("<qqqBBBBHHIIqqqq", 1, 20130315, 0, 0, 0, 1, 0, mode, 0, blocksize, 0, total, count, 0, iblocks)
    img[0 : len(hdr)] = hdr
    for i in range(count):
        b, n, data = content[i]
        img[b * blocksize : b * blocksize + len(data)] = data
        is_dir = order[i] in dirs or order[i] == "<super>"
        flags = 1 if i in compressed_inos else 0
        size = len(data) if not is_dir else n * blocksize
        node = struct.pack("<HHIqq", 0x4000 if is_dir else 0x8000, 1, flags, size, size) + b"\0" * 0x48
        node += struct.pack("<I", n)
        assert len(node) == 0x64
        if signed:
            ptrs = [b + k for k in range(n)]
            db = ptrs[:12] + [-1] * (12 - min(12, n))
            ib = [-1] * 5
            rest = ptrs[12:]
            extra = bytearray()
            if rest:
                # one single-indirect block of (signature, block) pairs, put at the end
                per_ib = blocksize // 36
                assert len(rest) <= per_ib
                ib[0] = total
                extra = bytearray(blocksize)
                for k, p in enumerate(rest):
                    struct.pack_into("<i", extra, k * 36 + 32, p)
                img += extra
                total += 1
            for v in db + ib:
                node += b"\0" * 32 + struct.pack("<i", v)
        else:
            for v in [b] + [-1] * 11 + [-1] * 5:
                node += struct.pack("<i", v)
        assert len(node) == isz
        at = blocksize * (1 + i // per) + isz * (i % per)
        img[at : at + isz] = node
    struct.pack_into("<q", img, 0x28, total)
    return img, ino


# Inner image, unsigned and not encrypted, then compressed.
inner, _ = pfs_image(0x10000, False, files)
bs = 0x10000
nblocks = len(inner) // bs
blocks = []
for i in range(nblocks):
    raw = bytes(inner[i * bs : (i + 1) * bs])
    z = zlib.compress(raw, 9)
    blocks.append(z if len(z) < bs else raw)
table_at = 0x30
data_at = (table_at + 8 * (nblocks + 1) + 0xF) & ~0xF
offsets, at = [], data_at
for b in blocks:
    offsets.append(at)
    at += len(b)
offsets.append(at)
pfsc = bytearray(struct.pack("<iiiiqqQq", 0x43534650, 0, 6, bs, bs, table_at, data_at, len(inner)))
pfsc += b"\0" * (table_at - len(pfsc))
pfsc += b"".join(struct.pack("<q", o) for o in offsets)
pfsc += b"\0" * (data_at - len(pfsc))
pfsc += b"".join(blocks)

# Outer image: signed, small blocks (so pfs_image.dat needs an indirect block), encrypted.
outer_bs = 0x1000
outer, oino = pfs_image(outer_bs, True, {"pfs_image.dat": bytes(pfsc)}, compressed_inos={2})
seed = rbytes(16)
mode = struct.unpack_from("<H", outer, 0x1C)[0] | 4
struct.pack_into("<H", outer, 0x1C, mode)
outer[0x370:0x380] = seed
ekpfs = rbytes(32)
pfs_flags = 0x2000000000000000  # the newer key derivation
k = hmac.new(ekpfs, seed, hashlib.sha256).digest()
key = hmac.new(k, struct.pack("<I", 1) + seed, hashlib.sha256).digest()
tweak_key, data_key = key[:16], key[16:]
for s in range(outer_bs // 0x1000, len(outer) // 0x1000):  # from the first block on
    enc = Cipher(algorithms.AES(data_key + tweak_key), modes.XTS(struct.pack("<Q", s) + b"\0" * 8)).encryptor()
    outer[s * 0x1000 : (s + 1) * 0x1000] = enc.update(bytes(outer[s * 0x1000 : (s + 1) * 0x1000])) + enc.finalize()


def sfo(pairs):
    keys, data, ents = b"", b"", b""
    for k, v in pairs:
        v = v.encode() + b"\0"
        ln = (len(v) + 3) // 4 * 4
        ents += struct.pack("<HHIII", len(keys), 0x0204, len(v), ln, len(data))
        keys += k.encode() + b"\0"
        data += v + b"\0" * (ln - len(v))
    keys += b"\0" * (-len(keys) % 4)
    kt = 20 + len(ents)
    return b"\0PSF" + struct.pack("<IIII", 0x101, kt, kt + len(keys), len(pairs)) + ents + keys + data


content_id, category, title, ver = {
    "game": ("UP0700-CUSA12776_00-HOKUTOGAAPP00000", "gd", "Test Game", "01.00"),
    "update": ("UP0700-CUSA12776_00-HOKUTOGAAPP00000", "gp", "Test Game", "01.05"),
    "dlc": ("UP0700-CUSA12776_00-HOKUTOGADLC00001", "ac", "Test Costume", "01.00"),
}[kind]
param = sfo([("APP_VER", ver), ("CATEGORY", category), ("CONTENT_ID", content_id), ("TITLE", title),
             ("TITLE_ID", "CUSA12776")])
icon = b"\x89PNG fake icon" * 10
license_info = rbytes(48)
dk3 = rbytes(32)

entry_keys = bytearray(2048)
entry_keys[32 * 8 + 256 * 3 : 32 * 8 + 256 * 4] = dk3key.encrypt(dk3, padding.PKCS1v15())

# Entries: (id, flags1, flags2, data); the image key and license.info are AES-sealed below.
entries = [
    [0x10, 0, 0, bytes(entry_keys)],
    [0x20, 0x80000000, 3 << 12, None],
    [0x401, 0x80000000, 3 << 12, None],
    [0x1000, 0, 0, param],
    [0x1200, 0, 0, icon],
]
table_at = 0x2000
data_at = table_at + 32 * len(entries)
pos = data_at
for e in entries:
    e.append(pos)
    pos += (len(e[3]) if e[3] is not None else (256 if e[0] == 0x20 else len(license_info)))
    pos = (pos + 15) & ~15
pfs_at = (pos + 0xFFFF) & ~0xFFFF


def row(e):
    size = len(e[3]) if e[3] is not None else (256 if e[0] == 0x20 else len(license_info))
    return struct.pack(">IIIIII", e[0], 0, e[1], e[2], e[4], size) + b"\0" * 8


def seal(e, plain):
    iv_key = hashlib.sha256(row(e) + dk3).digest()
    c = Cipher(algorithms.AES(iv_key[16:]), modes.CBC(iv_key[:16])).encryptor()
    return c.update(plain) + c.finalize()


for e in entries:
    if e[0] == 0x20:
        e[3] = seal(e, fake.encrypt(ekpfs, padding.PKCS1v15()))
    if e[0] == 0x401:
        e[3] = seal(e, license_info)

pkg = bytearray(pfs_at + len(outer))
struct.pack_into(">IIIIIHHII", pkg, 0, 0x7F434E54, 0, 0, 0, len(entries), 0, len(entries), table_at, 0)
pkg[0x40 : 0x40 + len(content_id)] = content_id.encode()
struct.pack_into(">IQQQ", pkg, 0x404, 1, pfs_flags, pfs_at, len(outer))
for i, e in enumerate(entries):
    pkg[table_at + 32 * i : table_at + 32 * (i + 1)] = row(e)
    pkg[e[4] : e[4] + len(e[3])] = e[3]
pkg[pfs_at:] = outer
open(out_pkg, "wb").write(pkg)

for path, data in list(files.items()) + [
    ("sce_sys/param.sfo", param),
    ("sce_sys/icon0.png", icon),
    ("sce_sys/license.info", license_info),
]:
    p = os.path.join(expected, path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)
