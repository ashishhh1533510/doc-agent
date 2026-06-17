"""
Deterministic extractor parity tests — no pytest, no LLM, no network.

Each case feeds a synthetic source string straight to a language extractor and
asserts the FOUR signals the HLD pipeline actually consumes (see
architecture_model.py): per-file `routes`, per-class `is_db_model`, entrypoint,
and behavioral content. The goal is parity: every supported language must light
up routes + persistence for its *idiomatic* web/ORM patterns, not just the
decorator/annotation ones.

Run:  ./venv/Scripts/python.exe tests/test_extractor_hld_signals.py
Exit code is non-zero if any case fails.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from doc_agent.tools.extractors.python_extractor import extract_from_python_source
from doc_agent.tools.extractors.ts_extractor import extract_from_typescript_source
from doc_agent.tools.extractors.java_extractor import extract_from_java_source
from doc_agent.tools.extractors.csharp_extractor import extract_from_csharp_source


# --------------------------------------------------------------------------- #
# tiny test harness
# --------------------------------------------------------------------------- #
_FAILURES: list[str] = []
_PASS = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global _PASS
    if cond:
        _PASS += 1
    else:
        _FAILURES.append(f"{label}" + (f"  ({detail})" if detail else ""))


def route_methods_paths(facts: dict) -> set:
    return {(r["method"], r["path"]) for r in facts.get("routes", [])}


def db_model_names(facts: dict) -> set:
    return {c["name"] for c in facts.get("classes", []) if c.get("is_db_model")}


# --------------------------------------------------------------------------- #
# JS / TS  (the NodeGoat blocker)
# --------------------------------------------------------------------------- #
def test_js_express_nested_routes():
    src = """
const express = require('express');
module.exports = function(app) {
    app.get('/', function (req, res) { res.render('home'); });
    app.post('/login', sessionHandler.handleLoginRequest);
    router.put('/users/:id', users.update);
};
"""
    f = extract_from_typescript_source(src, "routes/index.js", "javascript")
    rp = route_methods_paths(f)
    check("js/express nested GET /", ("GET", "/") in rp, str(rp))
    check("js/express nested POST /login", ("POST", "/login") in rp, str(rp))
    check("js/express nested PUT /users/:id", ("PUT", "/users/:id") in rp, str(rp))


def test_js_express_no_false_positive():
    # Map.get / cache.get have ONE argument and must not be mistaken for routes.
    src = """
const cache = new Map();
function load(key) {
    const v = cache.get(key);
    store.set('k', v);
    return v;
}
"""
    f = extract_from_typescript_source(src, "lib/cache.js", "javascript")
    check("js no false-positive route from Map.get", route_methods_paths(f) == set(),
          str(route_methods_paths(f)))


def test_js_mongoose_model():
    src = """
const mongoose = require('mongoose');
const userSchema = new mongoose.Schema({ name: String, email: String });
const User = mongoose.model('User', userSchema);
"""
    f = extract_from_typescript_source(src, "models/user.js", "javascript")
    check("js/mongoose User is_db_model", "User" in db_model_names(f), str(db_model_names(f)))


def test_js_sequelize_define():
    src = """
const User = sequelize.define('User', { name: DataTypes.STRING });
"""
    f = extract_from_typescript_source(src, "models/user.js", "javascript")
    check("js/sequelize define User is_db_model", "User" in db_model_names(f),
          str(db_model_names(f)))


def test_js_sequelize_class():
    src = """
class Product extends Model {}
Product.init({ name: DataTypes.STRING }, { sequelize });
"""
    f = extract_from_typescript_source(src, "models/product.js", "javascript")
    check("js/sequelize class Product is_db_model", "Product" in db_model_names(f),
          str(db_model_names(f)))


def test_js_raw_driver_dao():
    # NodeGoat uses the native MongoDB driver in DAO modules (no ORM at all).
    src = """
function UserDAO(db) {
    this.users = db.collection('users');
    this.getUserById = function (id, cb) { this.users.findOne({ _id: id }, cb); };
}
module.exports.UserDAO = UserDAO;
"""
    f = extract_from_typescript_source(src, "data/user-dao.js", "javascript")
    check("js/raw-driver collection -> persistence signal",
          len(db_model_names(f)) >= 1, str(db_model_names(f)))


def test_js_nestjs_still_works():
    # GREEN guard: decorator path must not regress.
    src = """
@Controller('cats')
export class CatsController {
    @Get(':id')
    findOne(id: string) { return id; }
}
@Entity()
export class Cat { name: string; }
"""
    f = extract_from_typescript_source(src, "cats.controller.ts", "typescript")
    check("js/nestjs route still detected", ("GET", "/cats/:id") in route_methods_paths(f),
          str(route_methods_paths(f)))
    check("js/typeorm entity still detected", "Cat" in db_model_names(f), str(db_model_names(f)))


# --------------------------------------------------------------------------- #
# Python
# --------------------------------------------------------------------------- #
def test_py_flask_route():
    src = """
from flask import Flask, Blueprint
app = Flask(__name__)
bp = Blueprint('api', __name__)

@app.route('/health', methods=['GET'])
def health():
    return 'ok'

@app.route('/users', methods=['POST'])
def create_user():
    return 'created'

@bp.route('/items')
def items():
    return 'items'
"""
    f = extract_from_python_source(src, "app.py")
    rp = route_methods_paths(f)
    check("py/flask GET /health", ("GET", "/health") in rp, str(rp))
    check("py/flask POST /users", ("POST", "/users") in rp, str(rp))
    check("py/flask default-GET /items", ("GET", "/items") in rp, str(rp))


def test_py_django_urlpatterns():
    src = """
from django.urls import path, re_path
from . import views

urlpatterns = [
    path('home/', views.home),
    path('users/<int:pk>/', views.user_detail),
    re_path(r'^legacy/$', views.legacy),
]
"""
    f = extract_from_python_source(src, "urls.py")
    paths = {r["path"] for r in f.get("routes", [])}
    check("py/django path home/", "home/" in paths, str(paths))
    check("py/django path users/<int:pk>/", "users/<int:pk>/" in paths, str(paths))
    check("py/django re_path legacy", any("legacy" in p for p in paths), str(paths))


def test_py_fastapi_still_works():
    # GREEN guard.
    src = """
from fastapi import APIRouter
router = APIRouter()

@router.get('/ping')
def ping():
    return 'pong'
"""
    f = extract_from_python_source(src, "api.py")
    check("py/fastapi route still detected", ("GET", "/ping") in route_methods_paths(f),
          str(route_methods_paths(f)))


def test_py_sqlalchemy_still_works():
    src = """
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = 'users'
    def name(self): return 'x'
"""
    f = extract_from_python_source(src, "models.py")
    check("py/sqlalchemy User is_db_model", "User" in db_model_names(f), str(db_model_names(f)))


# --------------------------------------------------------------------------- #
# Java
# --------------------------------------------------------------------------- #
def test_java_jaxrs_routes():
    src = """
package com.example;
import javax.ws.rs.GET;
import javax.ws.rs.POST;
import javax.ws.rs.Path;

@Path("/api/items")
public class ItemResource {
    @GET
    @Path("/{id}")
    public String get(String id) { return id; }

    @POST
    public String create() { return "ok"; }
}
"""
    f = extract_from_java_source(src, "ItemResource.java")
    rp = route_methods_paths(f)
    check("java/jaxrs GET /api/items/{id}", ("GET", "/api/items/{id}") in rp, str(rp))
    check("java/jaxrs POST /api/items", ("POST", "/api/items") in rp, str(rp))


def test_java_spring_still_works():
    # GREEN guard.
    src = """
package com.example;
import org.springframework.web.bind.annotation.*;
import javax.persistence.Entity;

@RestController
@RequestMapping("/cats")
public class CatController {
    @GetMapping("/{id}")
    public String get(String id) { return id; }
}

@Entity
class Cat { Long id; }
"""
    f = extract_from_java_source(src, "CatController.java")
    check("java/spring route still detected", ("GET", "/cats/{id}") in route_methods_paths(f),
          str(route_methods_paths(f)))
    check("java/jpa entity still detected", "Cat" in db_model_names(f), str(db_model_names(f)))


# --------------------------------------------------------------------------- #
# C#  (already at parity — lock it in)
# --------------------------------------------------------------------------- #
def test_csharp_still_works():
    src = """
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

[Route("api/[controller]")]
public class ProductsController : ControllerBase
{
    [HttpGet]
    public string List() => "x";
}

public class AppDb : DbContext
{
    public DbSet<Product> Products { get; set; }
}
"""
    f = extract_from_csharp_source(src, "ProductsController.cs")
    check("csharp route still detected",
          any(m == "GET" and p.lower().startswith("/api/products") for m, p in route_methods_paths(f)),
          str(route_methods_paths(f)))
    check("csharp dbcontext still detected", "AppDb" in db_model_names(f), str(db_model_names(f)))


# --------------------------------------------------------------------------- #
def main():
    cases = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for c in cases:
        try:
            c()
        except Exception as e:  # an extractor crash is itself a failure
            _FAILURES.append(f"{c.__name__} raised {type(e).__name__}: {e}")
    print(f"\n{_PASS} checks passed, {len(_FAILURES)} failed")
    for fail in _FAILURES:
        print(f"  FAIL  {fail}")
    return 1 if _FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
