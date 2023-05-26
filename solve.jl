function solve(Ω::Matrix{Int}, data::Vector{<:Array{<:Real}}; 
    β::Int=3, maxiter::Int=20000, verbose::Bool=true) 	

N = size(Ω,1)
length(data) == N || error("number of rows of multiinds and number of marginals missmatch!\n")

support = map(mu -> findall(mu[:] .> 0), data)
ell = length.(support)

cfun = makeCostfunction(WSplines())

# SELECT OTPMIZER
model = Model(GLPK.Optimizer)
# model = Model(Gurobi.Optimizer)
# model = Model(optimizer_with_attributes(Mosek.Optimizer, "QUIET" => true))
# model = direct_model(GLPK.Optimizer())
# model = direct_model(Gurobi.Optimizer())
# model = direct_model(MosekTools.Optimizer("QUIET" => true))

# setup model
JuMP.@variable(model, γ[1:size(Ω,2)] >= 0)
JuMP.@constraint(model, marginal[k = 1:length(data), j in 1:length(data[k])],
	sum(γ[findall(Ω[k,:] .== j)]) == data[k][j]
    )
JuMP.@objective(model, Min,
	sum(γ .* [cfun(r[:]) for r in eachcol(Ω)])
    )

# verify feasibility of initial plan
JuMP.optimize!(model)
JuMP.termination_status(model) === MOI.OPTIMAL || error("Failed solving with given initial plan: ",JuMP.termination_status(model))

verbose && print("Initial reduced problem solved! Starting GenCol...\n")

costvec = [objective_value(model)]
iter = 0
while iter < maxiter
	iter += 1
	
	# create child
	Iactive = findall(value.(γ) .> 0)
  potentials = dual.(marginal)
	child, samples = create_child(Ω[:,Iactive],potentials,support,cfun; tol=2*eps())
  child === nothing && break                              # termination criterion
    
    # add child to model
    push!(γ, @variable(model, lower_bound = 0))
    set_objective_coefficient(model, γ[end], cfun(child))
    for k in eachindex(data)
    	set_normalized_coefficient(marginal[k,child[k]], γ[end], 1)
    end
    Ω = hcat(Ω,child); dΩ = size(Ω,2)

    # reoptimize model
    optimize!(model)
    push!(costvec, objective_value(model))
    
    # Tail clearing
    if dΩ > β*sum(ell)
        Iactive = findall(value.(γ) .> 0)
      	verbose && print("Tail clearing! Iter: ",lpad(string(iter),textwidth(string(maxiter))),", |spt(γ)| = ",length(Iactive),", |Ω| = $dΩ")
      	basis = _findbasis(model,γ) # basis can be bigger than active space!
        inactive = setdiff(1:dΩ,basis)
        Ω = Ω[:, setdiff(1:dΩ, inactive[1:length(basis)])]
        delete(model,γ[inactive[1:length(basis)]])
        γ = γ[setdiff(1:dΩ, inactive[1:length(basis)])]
        dΩ = size(Ω,2);
        verbose && print(" -> $dΩ \n")
        optimize!(model) # reoptimize
    end
end

print( IOContext(stdout, :compact => true),
	"GenCol stopped after $iter iterations. Current optimal cost: ",objective_value(model),"\n")

# build output
res = (model = model, multiinds = Ω, gamma = γ, u = marginal, costvec = costvec)

return res
end
