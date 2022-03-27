mutable struct Solver{T}
    problem::ProblemData{T} 
    methods::ProblemMethods 
    data::SolverData{T}
    variables::Vector{T} 
    candidate::Vector{T}
    indices::Indices
    dimensions::Dimensions
    linear_solver::LinearSolver
    central_path::Vector{T} 
    penalty::Vector{T}
    dual::Vector{T}
    options::Options{T}

    lagrangian_gradient::Vector{T}
    
    merit::T                            # merit objective value
    merit_candidate::T                           # next merit objective value
    merit_gradient::Vector{T}                   # gradient of merit objective

    c::Vector{T}                    # constraint values
    c_soc::Vector{T}
    c_tmp::Vector{T}

    # Line search values
    step_size::T
    dual_step_size::T
    maximum_step_size::T
    minimum_step_size::T

    # Regularization
    regularization::Vector{T}
    primal_regularization::T
    primal_regularization_last::T
    dual_regularization::T

    # Constraint violation
    constraint_violation::T          # 1-norm of constraint violation
    constraint_violation_candidate::T
    min_constraint_violation::T
    max_constraint_violation::T
    constraint_violation_correction::T

    fraction_to_boundary::T
    filter::Vector{Tuple}

    # iteration counts
    outer_iteration::Int   # central path iteration (outer loop)
    residual_iteration::Int   # merit problem iteration
    line_search_iteration::Int   # line search
    soc_iteration::Int   # second order corrections

    failures::Int

    scaling_objective::T
    scaling_constraints::SparseMatrixCSC{T,Int}
end

function Solver(x, methods, num_variables, num_equality, num_inequality; 
    options=Options())

    # problem data
    p_data = ProblemData(num_variables, num_equality, num_inequality)

    # solver data
    s_data = SolverData(num_variables, num_equality, num_inequality)

    # indices
    idx = Indices(num_variables, num_equality, num_inequality)

    # dimensions 
    dim = Dimensions(num_variables, num_equality, num_inequality)

    # variables 
    variables = zeros(dim.total) 
    variables[1:dim.variables] = copy(x)
    variables[dim.symmetric .+ (1:(2 * dim.inequality))] .= 1.0

    candidate = zeros(dim.total)

    # interior-point 
    central_path = [1.0] 

    # augmented Lagrangian 
    penalty = [1.0] 
    dual = zeros(num_equality) 

    # linear solver TODO: constructor
    
    problem!(p_data, methods, idx, variables,
        gradient=true,
        constraint=true,
        jacobian=true,
        hessian=true)
    matrix!(s_data, p_data, idx, variables, central_path, penalty, dual, 1.0, 1.0)
    matrix_symmetric!(s_data.matrix_symmetric, s_data.matrix, idx)
    # linear_solver = ldl_solver(s_data.matrix_symmetric)

    F = qdldl(s_data.matrix_symmetric)
    inertia = Inertia(0, 0, 0)
    linear_solver = QDLDLSolver(F, inertia)

    #####

    # scaling
    scaling_objective = objective_gradient_scaling(options.scaling_tolerance, [p_data.objective_gradient; zeros(num_inequality)])
    scaling_constraints = constraint_scaling(options.scaling_tolerance, [p_data.equality_jacobian; p_data.inequality_jacobian], dim.constraints)
    
    # fraction to boundary
    fraction = fraction_to_boundary(central_path[1], options.min_fraction_to_boundary)

    # Lagrangian
    lagrangian_gradient = zeros(dim.primal) 

    # merit 
    merit = 0.0
    merit_candidate = 0.0
    merit_gradient = zeros(dim.primal)

    c = zeros(dim.constraints)
    c_soc = zeros(dim.constraints)
    c_tmp = zeros(dim.constraints)

    c .= [p_data.equality; p_data.inequality - variables[idx.slack_primal]]
    c .= scaling_constraints * c

    step_size = 1.0
    dual_step_size = 1.0
    maximum_step_size = 1.0
    minimum_step_size = 1.0

    regularization = zeros(dim.total)
    primal_regularization = 0.
    primal_regularization_last = 0.
    dual_regularization = 0.

    filter = Tuple[]

    outer_iteration = 0
    residual_iteration = 0
    line_search_iteration = 0
    soc_iteration = 0

    failures = 0

    constraint_violation = norm(c, 1)
    constraint_violation_candidate = copy(constraint_violation)
    min_constraint_violation = initialize_min_constraint_violation(constraint_violation)
    max_constraint_violation = initialize_max_constraint_violation(constraint_violation)

    constraint_violation_correction = 0.

    #####

    Solver(
        p_data, 
        methods, 
        s_data,
        variables,
        candidate, 
        idx, 
        dim,
        linear_solver,
        central_path, 
        penalty, 
        dual,
        options,
        lagrangian_gradient,
        merit,merit_candidate,merit_gradient,
        c,c_soc,c_tmp,
        step_size,dual_step_size,maximum_step_size,minimum_step_size,
        regularization,primal_regularization,primal_regularization_last,dual_regularization,
        constraint_violation,constraint_violation_candidate,min_constraint_violation,max_constraint_violation,constraint_violation_correction,
        fraction,
        filter,
        outer_iteration,residual_iteration,line_search_iteration,soc_iteration,
        failures,
        scaling_objective, scaling_constraints,
    )
end
