
function Model(nlp)
    nlp.meta.lcon
    nlp.meta.ucon

    n = nlp.meta.nvar
    m = nlp.meta.ncon

    # println("n: $n\nm: $m")

    x0 = nlp.meta.x0
    xL = nlp.meta.lvar
    xU = nlp.meta.uvar

    f_func(x,model) = obj(nlp,x)
    function ∇f_func!(∇f,x,model)
        grad!(nlp,x,∇f)
        return nothing
    end
    function ∇²f_func!(∇²f,x,model)
        ∇²f .= hess(nlp,x)
        return nothing
    end

    function c_func!(c,x,model)
        cons!(nlp,x,c)
        return nothing
    end
    function ∇c_func!(∇c,x,model)
        ∇c .= jac(nlp,x)
        return nothing
    end
    function ∇²cy_func!(∇²cy,x,y,model)
        ∇²cy .= hess(nlp,x,y) - hess(nlp,x)
        return nothing
    end

    model = Model(n,m,xL,xU,f_func,∇f_func!,∇²f_func!,c_func!,∇c_func!,∇²cy_func!,cA_idx=ones(Bool,m))
end

function Ipopt.createProblem(nlp)
    function eval_f(x)
        obj(nlp, x)
    end
    function eval_grad_f(x, grad_f)
        grad!(nlp, x, grad_f)
    end
    function eval_h(x, mode, rows, cols, obj_factor, lambda, values)
        if mode == :Structure
            hess_structure!(nlp, rows, cols)
        else
            hess_coord!(nlp, x, lambda, values, obj_weight=obj_factor)
        end
    end
    function eval_g(x, g)
        cons!(nlp, x, g)
    end
    function eval_jac_g(x, mode, rows, cols, values)
        if mode == :Structure
            jac_structure!(nlp, rows, cols)
        else
            jac_coord!(nlp, x, values)
        end
    end

    n = nlp.meta.nvar
    m = nlp.meta.ncon
    x_L = nlp.meta.lvar
    x_U = nlp.meta.uvar
    g_L = nlp.meta.lcon
    g_U = nlp.meta.ucon

    prob = createProblem(n, x_L, x_U, m, g_L, g_U, nlp.meta.nnzj, nlp.meta.nnzh,
        eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h)
    prob.x .= nlp.meta.x0
    prob
end
