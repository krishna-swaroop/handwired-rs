// #[handwired::step]: keeps a plain Rust function as written, and adds two
// functions next to it:
//
// - `<name>_in<D: Domain>(domain, params...)`: the same body, written against
//   handwired's `Domain` vocabulary, so it can run as numbers (`Native`) or
//   build a network (`CircuitDomain`). See crates/handwired/FRONTEND.md.
// - `<name>_circuit()`: the network, with one source cell per parameter,
//   declared with the parameter's `#[range(lo, hi)]`, on the integer grid for
//   integer and bool parameters.
//
// The subset is M0's: literals, parameters, `let` and reassignment, `+ - *`,
// comparisons, `&& || !`, unary `-`, and `if`/`else`. Anything else is a
// compile error at its span.

use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as Tokens};
use quote::{format_ident, quote, quote_spanned};
use syn::spanned::Spanned;
use syn::{BinOp, Block, Expr, FnArg, ItemFn, Lit, Pat, Stmt, Type, UnOp, parse_macro_input};

#[proc_macro_attribute]
pub fn step(attribute: TokenStream, item: TokenStream) -> TokenStream {
    if !attribute.is_empty() {
        let error = syn::Error::new(Span::call_site(), "#[handwired::step] takes no arguments");
        return error.to_compile_error().into();
    }
    let function = parse_macro_input!(item as ItemFn);
    expand(function).unwrap_or_else(|error| error.to_compile_error()).into()
}

struct Parameter {
    name: syn::Ident,
    minimum: f64,
    maximum: f64,
    integer: bool,
}

fn expand(mut function: ItemFn) -> syn::Result<Tokens> {
    let signature = &function.sig;
    if !signature.generics.params.is_empty() || signature.asyncness.is_some() {
        return Err(syn::Error::new(signature.span(), "#[handwired::step] needs a plain function without generics"));
    }
    let mut parameters = Vec::new();
    for input in function.sig.inputs.iter_mut() {
        let FnArg::Typed(typed) = input else {
            return Err(syn::Error::new(input.span(), "#[handwired::step] functions take no `self`"));
        };
        let Pat::Ident(pattern) = &*typed.pat else {
            return Err(syn::Error::new(typed.pat.span(), "a step parameter must be a plain name"));
        };
        let name = pattern.ident.clone();
        let kind = number_kind(&typed.ty)?;
        let range_index = typed.attrs.iter().position(|attribute| attribute.path().is_ident("range"));
        let range = match range_index {
            Some(index) => Some(parse_range(&typed.attrs.remove(index))?),
            None => None,
        };
        let (minimum, maximum) = match (range, kind) {
            (Some(range), _) => range,
            (None, NumberKind::Bool) => (0.0, 1.0),
            (None, _) => {
                return Err(syn::Error::new(
                    name.span(),
                    format!("`{name}` needs a declared range, e.g. #[range(0, 100)] {name}: ..."),
                ));
            }
        };
        if minimum > maximum {
            return Err(syn::Error::new(name.span(), "a range's minimum is above its maximum"));
        }
        parameters.push(Parameter { name, minimum, maximum, integer: kind != NumberKind::Float });
    }

    let body = translate_block(&function.block)?;
    let visibility = &function.vis;
    let name = &function.sig.ident;
    let name_in = format_ident!("{}_in", name);
    let name_circuit = format_ident!("{}_circuit", name);
    let names: Vec<&syn::Ident> = parameters.iter().map(|parameter| &parameter.name).collect();
    let count = parameters.len();
    let declarations = parameters.iter().enumerate().map(|(index, parameter)| {
        let (minimum, maximum) = (parameter.minimum, parameter.maximum);
        let grid = parameter.integer.then(|| quote! { circuit.assume_grid(inputs[#index], 1.0); });
        quote! {
            circuit.assume_range(inputs[#index], #minimum, #maximum);
            #grid
        }
    });
    let arguments = (0..count).map(|index| quote! { inputs[#index] });
    let documentation_in = format!("`{name}` written against `handwired::Domain`; see `#[handwired::step]`.");
    let documentation_circuit = format!("`{name}` as a network: one source cell per parameter, in order.");
    Ok(quote! {
        #function

        #[doc = #documentation_in]
        #[allow(unused_mut, clippy::let_and_return)]
        #visibility fn #name_in<D: ::handwired::Domain>(__domain: &mut D, #(#names: D::Value),*) -> D::Value #body

        #[doc = #documentation_circuit]
        #visibility fn #name_circuit() -> ::handwired::StepCircuit {
            let mut circuit = ::handwired::Circuit::new(#count);
            let inputs: ::std::vec::Vec<::handwired::Wire> = (0..#count).map(|index| circuit.source(index)).collect();
            #(#declarations)*
            let output = {
                let mut domain = ::handwired::CircuitDomain { circuit: &mut circuit };
                #name_in(&mut domain, #(#arguments),*)
            };
            ::handwired::StepCircuit { circuit, inputs, output }
        }
    })
}

#[derive(Copy, Clone, PartialEq)]
enum NumberKind {
    Integer,
    Float,
    Bool,
}

fn number_kind(ty: &Type) -> syn::Result<NumberKind> {
    if let Type::Path(path) = ty
        && let Some(ident) = path.path.get_ident()
    {
        let name = ident.to_string();
        match name.as_str() {
            "i8" | "i16" | "i32" | "i64" | "isize" | "u8" | "u16" | "u32" | "u64" | "usize" => return Ok(NumberKind::Integer),
            "f32" | "f64" => return Ok(NumberKind::Float),
            "bool" => return Ok(NumberKind::Bool),
            _ => {}
        }
    }
    Err(syn::Error::new(ty.span(), "a step parameter must be an integer, f32, f64 or bool"))
}

fn parse_range(attribute: &syn::Attribute) -> syn::Result<(f64, f64)> {
    let bounds = attribute.parse_args_with(syn::punctuated::Punctuated::<Expr, syn::Token![,]>::parse_terminated)?;
    let values = bounds.iter().map(constant_number).collect::<syn::Result<Vec<f64>>>()?;
    match values[..] {
        [minimum, maximum] => Ok((minimum, maximum)),
        _ => Err(syn::Error::new(attribute.span(), "#[range] takes two numbers: #[range(minimum, maximum)]")),
    }
}

fn constant_number(expression: &Expr) -> syn::Result<f64> {
    match expression {
        Expr::Lit(literal) => literal_number(&literal.lit),
        Expr::Unary(unary) if matches!(unary.op, UnOp::Neg(_)) => Ok(-constant_number(&unary.expr)?),
        _ => Err(syn::Error::new(expression.span(), "expected a number")),
    }
}

fn literal_number(literal: &Lit) -> syn::Result<f64> {
    match literal {
        Lit::Int(integer) => integer.base10_parse::<f64>(),
        Lit::Float(float) => float.base10_parse::<f64>(),
        Lit::Bool(boolean) => Ok(if boolean.value { 1.0 } else { 0.0 }),
        _ => Err(syn::Error::new(literal.span(), "expected a number or bool literal")),
    }
}

fn unsupported(span: Span, what: &str) -> syn::Error {
    syn::Error::new(span, format!("#[handwired::step] can't translate {what} yet; see crates/handwired/FRONTEND.md for the subset"))
}

fn translate_block(block: &Block) -> syn::Result<Tokens> {
    let mut statements = Vec::new();
    let mut value = None;
    for (index, statement) in block.stmts.iter().enumerate() {
        let last = index + 1 == block.stmts.len();
        match statement {
            Stmt::Local(local) => {
                let pattern = match &local.pat {
                    Pat::Ident(pattern) => pattern,
                    Pat::Type(typed) => match &*typed.pat {
                        Pat::Ident(pattern) => pattern,
                        other => return Err(unsupported(other.span(), "this pattern")),
                    },
                    other => return Err(unsupported(other.span(), "this pattern")),
                };
                let Some(init) = &local.init else {
                    return Err(unsupported(local.span(), "a `let` without a value"));
                };
                if init.diverge.is_some() {
                    return Err(unsupported(local.span(), "`let ... else`"));
                }
                let mutability = &pattern.mutability;
                let ident = &pattern.ident;
                let expression = translate(&init.expr)?;
                statements.push(quote! { let #mutability #ident = #expression; });
            }
            Stmt::Expr(expression, None) if last => value = Some(translate(expression)?),
            Stmt::Expr(expression, Some(_)) => statements.push(translate_statement(expression)?),
            other => return Err(unsupported(other.span(), "this statement")),
        }
    }
    let Some(value) = value else {
        return Err(unsupported(block.span(), "a block that doesn't end in a value"));
    };
    Ok(quote! {{ #(#statements)* #value }})
}

// `x = e;` and `x += e;`, on a `let mut`.
fn translate_statement(expression: &Expr) -> syn::Result<Tokens> {
    let assigned = |target: &Expr| -> syn::Result<syn::Ident> {
        match target {
            Expr::Path(path) if path.path.get_ident().is_some() => Ok(path.path.get_ident().unwrap().clone()),
            other => Err(unsupported(other.span(), "an assignment to anything but a local")),
        }
    };
    match expression {
        Expr::Assign(assign) => {
            let target = assigned(&assign.left)?;
            let value = translate(&assign.right)?;
            Ok(quote! { #target = #value; })
        }
        Expr::Binary(binary) => {
            let method = match binary.op {
                BinOp::AddAssign(_) => quote! { add },
                BinOp::SubAssign(_) => quote! { sub },
                BinOp::MulAssign(_) => quote! { mul },
                _ => return Err(unsupported(binary.span(), "an expression statement")),
            };
            let target = assigned(&binary.left)?;
            let value = translate(&binary.right)?;
            Ok(quote! {{ let __right = #value; #target = __domain.#method(#target, __right); }})
        }
        other => Err(unsupported(other.span(), "an expression statement")),
    }
}

fn translate(expression: &Expr) -> syn::Result<Tokens> {
    let span = expression.span();
    match expression {
        Expr::Lit(literal) => {
            let value = literal_number(&literal.lit)?;
            Ok(quote_spanned! {span=> __domain.constant(#value) })
        }
        Expr::Path(path) => match path.path.get_ident() {
            Some(ident) => Ok(quote! { #ident }),
            None => Err(unsupported(span, "a path")),
        },
        Expr::Paren(paren) => translate(&paren.expr),
        Expr::Group(group) => translate(&group.expr),
        Expr::Block(block) if block.label.is_none() => translate_block(&block.block),
        Expr::Unary(unary) => {
            let operand = translate(&unary.expr)?;
            let method = match unary.op {
                UnOp::Neg(_) => quote! { neg },
                UnOp::Not(_) => quote! { not },
                _ => return Err(unsupported(span, "a dereference")),
            };
            Ok(quote_spanned! {span=> { let __operand = #operand; __domain.#method(__operand) } })
        }
        Expr::Binary(binary) => {
            let left = translate(&binary.left)?;
            let right = translate(&binary.right)?;
            let relation = |name: &str| {
                let variant = format_ident!("{}", name);
                quote! { __domain.compare(__left, ::handwired::Relation::#variant, __right) }
            };
            let operation = match binary.op {
                BinOp::Add(_) => quote! { __domain.add(__left, __right) },
                BinOp::Sub(_) => quote! { __domain.sub(__left, __right) },
                BinOp::Mul(_) => quote! { __domain.mul(__left, __right) },
                BinOp::Gt(_) => relation("Greater"),
                BinOp::Ge(_) => relation("AtLeast"),
                BinOp::Lt(_) => relation("Less"),
                BinOp::Le(_) => relation("AtMost"),
                BinOp::Eq(_) => relation("Equal"),
                BinOp::Ne(_) => {
                    let equal = relation("Equal");
                    quote! {{ let __equal = #equal; __domain.not(__equal) }}
                }
                // Both sides are pure, so evaluating both is the same as short-circuiting.
                BinOp::And(_) => quote! { __domain.and(__left, __right) },
                BinOp::Or(_) => quote! { __domain.or(__left, __right) },
                BinOp::Div(_) | BinOp::Rem(_) => return Err(unsupported(span, "division or remainder")),
                _ => return Err(unsupported(span, "this operator")),
            };
            Ok(quote_spanned! {span=> { let __left = #left; let __right = #right; #operation } })
        }
        Expr::If(conditional) => {
            let Some((_, otherwise)) = &conditional.else_branch else {
                return Err(unsupported(span, "an `if` without an `else`"));
            };
            if let Expr::Let(_) = &*conditional.cond {
                return Err(unsupported(conditional.cond.span(), "`if let`"));
            }
            let gate = translate(&conditional.cond)?;
            let on = translate_block(&conditional.then_branch)?;
            let off = translate(otherwise)?;
            // Both branches become part of the network; the gate picks one.
            Ok(quote_spanned! {span=> {
                let __gate = #gate;
                let __on = #on;
                let __off = #off;
                __domain.select(__gate, __off, __on)
            }})
        }
        Expr::Return(_) => Err(unsupported(span, "`return`; end the block with its value instead")),
        Expr::Call(_) | Expr::MethodCall(_) => Err(unsupported(span, "a call")),
        Expr::Index(_) => Err(unsupported(span, "indexing")),
        Expr::ForLoop(_) | Expr::While(_) | Expr::Loop(_) => Err(unsupported(span, "a loop")),
        Expr::Match(_) => Err(unsupported(span, "`match`")),
        Expr::Cast(_) => Err(unsupported(span, "a cast")),
        _ => Err(unsupported(span, "this expression")),
    }
}
