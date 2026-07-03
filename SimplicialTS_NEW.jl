module SimplicialTS_NEW


using Ripserer
using LinearAlgebra
using Combinatorics
using LoopVectorization
using PersistenceDiagrams
using Statistics
using DelimitedFiles


export simplicial_complex_mvts,z_score,compute_scaffold_fast_tri, normalize_t_d,compute_edgeweight,search_good_quadruplet,fix_violations_and_compute_scaffold,find_max_weight,compute_scaffold,compute_scaffold_fast,fix_violations_and_compute_qua_coherence, find_maximum_two_vecs!,create_data_structure,coherence_function,correction_for_coherence,create_simplicial_complex, fix_violations_and_compute_complexity, compute_complexity, load_data,load_data_mat,load_data_synthetic_kaneko, load_normaltxt, find_max_weight, compute_scaffold_fast, sliced_wasserstein


struct simplicial_complex_mvts
    raw_data::Matrix{Float64}
    num_ROI::Int64
    T::Int64
    ets_indexes::Dict{Int64,Vector{Int64}}
    ets_zscore::Matrix{Float64}#创造一个矩阵存储边信息，边的数量行，2列
    ets_max::Vector{Float64}
    triplets_indexes::Dict{Int64,Vector{Int64}}
    triplets_indexes_reverse::Dict{Tuple{Int64,Int64,Int64},Int64}
    triplets_zscore::Matrix{Float64}
    triplets_max::Vector{Float64}
    #四面体quadruplet
    quadruplet_indexes::Dict{Int64,Vector{Int64}}
    quadruplet_indexes_reverse::Dict{Tuple{Int64,Int64,Int64,Int64},Int64}
    quadruplet_zscore::Matrix{Float64}
    quadruplet_max::Vector{Float64}


end

function cityblock(x,y)

    return sum(abs.(x-y))
end

function z_score(data::Matrix{Float64}, N::Int64, T::Int64)::Matrix{Float64}
    zscore_raw=zeros((N,T))
    l=1
    @inbounds for row in eachrow(data)
        m=mean(row)
        s=std(row,mean=m,corrected=false)
        zscore_raw[l,:]=(row.-m)./s
        l+=1
    end
    return(zscore_raw)

end

#对时间导数矩阵里每个元素除以该行的标准差，进行一个标准化
function normalize_t_d(data::Matrix{Float64}, N::Int64, T::Int64)::Matrix{Float64}
    normalize_raw=zeros((N,T))
    l=1
    @inbounds for row in eachrow(data)
        m=mean(row)
        s=std(row,mean=m,corrected=false)
        normalize_raw[l,:]=row./s
        l+=1
    end
    #println(normalize_raw)
    return(normalize_raw)

end

function find_maximum_two_vecs!(vec1::Vector{Float64},vec2::Vector{Float64})
    @inbounds for i in eachindex(vec1)
        vec1[i]=max(abs(vec1[i]),abs(vec2[i]))
        end
end


function create_data_structure(data::Matrix{Float64})::simplicial_complex_mvts
    N,T=size(data)
    
    T_d = Matrix{Float64}(undef, N, T-1)
    #计算时间导数
    for i in 1:N
        for j in 1:T-1
           T_d[i,j] = data[i,j+1] -data[i,j]
        end
    end
    T_d = normalize_t_d(T_d,N,T-1)
    ##Compute the edges
    N_edges = div(N*(N-1),2)
    ets_zscore = similar(data,(N_edges, 2))
    ets_max=zeros(T-1)#similar(data,T)#长度为时间T的向量存储，每个时间点上权重最大的二元组
    ets_indexes=Dict{Int64, Vector{Int64}}()
    current_val=Vector{Float64}()
    l=1
    @inbounds for i in 1:N
         for j in  i+1:N
                @views current_val = T_d[i,:] .* T_d[j,:]#一个新的时间序列
                m = mean(current_val)
                s = std(current_val,mean=m,corrected=false,)
                ets_zscore[l,1] = m
                ets_zscore[l,2] = s#存储二元组的均值和方差
                find_maximum_two_vecs!(ets_max,(current_val .- m) ./ s)
                ets_indexes[l] = [i,j]#按照顺序记录下，每条边的索引。边映射到索引
                l+=1
        end
    end


    ##Compute the triplets
    N_triplets =binomial(N,3)
    triplets_zscore=similar(data,(N_triplets,2))
    triplets_max=zeros(T-1)
    triplets_indexes=Dict{Int64, Vector{Int64}}()
    triplets_indexes_reverse = Dict{Tuple{Int64,Int64,Int64}, Int64}()
    l=1
    @inbounds for i in 1:N
         for j in i+1:N
            for k in j+1:N
            @views current_val = T_d[i,:] .* T_d[j,:] .* T_d[k,:]
            m = mean(current_val)
            s = std(current_val,mean=m,corrected=false,)
            triplets_zscore[l,1] = m
            triplets_zscore[l,2] = s
            find_maximum_two_vecs!(triplets_max,(current_val .- m) ./ s)
            triplets_indexes[l]=[i,j,k]
            triplets_indexes_reverse[(i,j,k)]=l
            l+=1
            end
        end
    end


    ##Compute the quadruplet
    N_quadruplet =binomial(N,4)
    quadruplet_zscore=similar(data,(N_quadruplet,2))
    quadruplet_max=zeros(T-1)
    quadruplet_indexes=Dict{Int64,Vector{Int64}}()
    quadruplet_indexes_reverse=Dict{Tuple{Int64,Int64,Int64,Int64},Int64}()

    l=1
    @inbounds for i in 1:N
         for j in i+1:N
            for k in j+1:N
                for p in k+1:N
                @views current_val = T_d[i,:] .* T_d[j,:] .* T_d[k,:].* T_d[p,:]
                m = mean(current_val)
                s = std(current_val,mean=m,corrected=false,)
                quadruplet_zscore[l,1] = m
                quadruplet_zscore[l,2] = s
                find_maximum_two_vecs!(quadruplet_max,(current_val .- m) ./ s)
                quadruplet_indexes[l] = [i,j,k,p]
                quadruplet_indexes_reverse[(i,j,k,p)] = l
                l+=1
                end
            end
        end
    end
    data_simplex=simplicial_complex_mvts(T_d,N,T-1,ets_indexes,ets_zscore,ets_max,triplets_indexes,triplets_indexes_reverse,triplets_zscore,triplets_max,quadruplet_indexes,quadruplet_indexes_reverse,quadruplet_zscore,quadruplet_max)
    return(data_simplex)
end
#输入一个矩阵，输出一个data simplex，输出一个结构体
#此时这个结构体里的时间序列矩阵已经变成了标准化后的时间导数矩阵，N行T-1列
function coherence_function(vector::Vector{Float64})::Int64
    n = length(vector)
    temp = 0
    for el in vector
        temp += sign(el)
    end
    exponent = sign(n - abs(temp))
    res = (-1)^exponent
    return res
end

function correction_for_coherence(current_list_sign::Vector{Float64}, current_weight::Float64)::Float64
    # If the original signals are fully coherent, then the corresponding weight becomes positive, otherwise negative
    coherence = coherence_function(@views current_list_sign)
    # If all the signs are concordant then set the weight sign as positive, otherwise negative
    if coherence == 1
        weight_corrected = abs(current_weight)
    else
        weight_corrected = -abs(current_weight)
    end
    return weight_corrected
end
function create_simplicial_complex(simplicial_TS::simplicial_complex_mvts,t_current::Int64)
    #list_simplices_positive = Vector{Tuple{Vector{Int64}, Float64}}(undef, simplicial_TS.num_ROI + length(simplicial_TS.ets_indexes) + length(simplicial_TS.triplets_indexes) + length(simplicial_TS.quadruplet_indexes))
    #列表存储了所有正协同作用的单型，也就是权值大于0，但是每个脑区相对于前一个时间是增加
    #list_simplices_negative = Vector{Tuple{Vector{Int64}, Float64}}(undef, simplicial_TS.num_ROI + length(simplicial_TS.ets_indexes) + length(simplicial_TS.triplets_indexes) + length(simplicial_TS.quadruplet_indexes))
    #列表存储了所有负协同作用的单型，也就是权值大于于0，但是每个脑区相对于前一个时间是减少
    list_simplices_positive = Vector{Tuple{Vector{Int64}, Float64}}()
    list_simplices_negative = Vector{Tuple{Vector{Int64}, Float64}}()
    
    # compute the extremal weight
    m_weight = max(ceil(simplicial_TS.quadruplet_max[t_current]),ceil(simplicial_TS.triplets_max[t_current]), ceil(simplicial_TS.ets_max[t_current]))

    #add the nodes to the list of simplices
    @inbounds for i in 1:simplicial_TS.num_ROI
        push!(list_simplices_positive, ([i], m_weight))
        push!(list_simplices_negative, ([i], m_weight))
    end
    println("ok")
    edges_matrix_positive = zeros(90, 90)
    edges_matrix_negative = zeros(90, 90)
    edges_matrix_negative_negative = zeros(90, 90)
    # Adding the edges to the list of simplices
    j=simplicial_TS.num_ROI+1
    k=simplicial_TS.num_ROI+1
    @inbounds for (i, indexes_ij) in simplicial_TS.ets_indexes
        c_mean = simplicial_TS.ets_zscore[i,1]
        c_std  = simplicial_TS.ets_zscore[i,2]
        raw_data_i = simplicial_TS.raw_data[indexes_ij[1],t_current]
        raw_data_j = simplicial_TS.raw_data[indexes_ij[2],t_current]
        weight_current = (raw_data_i * raw_data_j - c_mean) / c_std
        list_of_signs_edges = [raw_data_i, raw_data_j]
        @views weight_current_corrected = correction_for_coherence(list_of_signs_edges, weight_current)
        if weight_current_corrected >= 0 
           if raw_data_i >= 0
                push!(list_simplices_positive, (indexes_ij, weight_current_corrected))
                
                edges_matrix_positive[indexes_ij[1], indexes_ij[2]] = weight_current_corrected
            else
                push!(list_simplices_negative, (indexes_ij, weight_current_corrected))
                
                edges_matrix_negative[indexes_ij[1], indexes_ij[2]] = weight_current_corrected
            end
        end 
        if weight_current_corrected < 0  
           edges_matrix_negative_negative[indexes_ij[1], indexes_ij[2]] = weight_current_corrected
        end
    end

    
    # Adding the triplets
    # Here I modify the signs of the weights, if it is fully coherent I assign a positive sign, otherwise negative
    @inbounds for (i,indexes_ijk) in simplicial_TS.triplets_indexes
        c_mean = simplicial_TS.triplets_zscore[i,1]
        c_std  = simplicial_TS.triplets_zscore[i,2]
        raw_data_i = simplicial_TS.raw_data[indexes_ijk[1],t_current]
        raw_data_j = simplicial_TS.raw_data[indexes_ijk[2],t_current]
        raw_data_k = simplicial_TS.raw_data[indexes_ijk[3],t_current]
        weight_current = (raw_data_i * raw_data_j * raw_data_k - c_mean) / c_std
        @views weight_current_corrected = correction_for_coherence([raw_data_i, raw_data_j,raw_data_k], weight_current)
        if weight_current_corrected >= 0 
            if raw_data_i >= 0
                push!(list_simplices_positive, (indexes_ijk, weight_current_corrected))
            else
                push!(list_simplices_negative, (indexes_ijk, weight_current_corrected))
            end
        end    
       
    end
   
    # Adding the quadruplet
    # Here I modify the signs of the weights, if it is fully coherent I assign a positive sign, otherwise negative
    @inbounds for (i,indexes_ijkp) in simplicial_TS.quadruplet_indexes
        c_mean = simplicial_TS.quadruplet_zscore[i,1]
        c_std  = simplicial_TS.quadruplet_zscore[i,2]
        raw_data_i = simplicial_TS.raw_data[indexes_ijkp[1],t_current]
        raw_data_j = simplicial_TS.raw_data[indexes_ijkp[2],t_current]
        raw_data_k = simplicial_TS.raw_data[indexes_ijkp[3],t_current]
        raw_data_p = simplicial_TS.raw_data[indexes_ijkp[4],t_current]
        weight_current = (raw_data_i * raw_data_j * raw_data_k * raw_data_p - c_mean) / c_std
        @views weight_current_corrected = correction_for_coherence([raw_data_i, raw_data_j,raw_data_k,raw_data_p], weight_current)
        if weight_current_corrected >= 0 
            if raw_data_i >= 0
                push!(list_simplices_positive, (indexes_ijkp, weight_current_corrected))
                j+=1
            else
                push!(list_simplices_negative, (indexes_ijkp, weight_current_corrected))
                k+=1
            end
        end    
    end

    println(j)
    println(k)
    return list_simplices_positive, list_simplices_negative,edges_matrix_positive,edges_matrix_negative,edges_matrix_negative_negative
end
#此时这个对于每个时间点，产生了一个包含所有边，三角形，四面体的单纯复形。

#下面这个函数用于产生每个时间点的空腔的持续同调H2的加权矩阵
function fix_violations_and_compute_scaffold(sorted_simplices::Vector{Tuple{Vector{Int64}, Float64}},t_current::Int,simplicial_TS::simplicial_complex_mvts)
    # Sorting the simplices in a descending order according to the weights
    
    sort!(sorted_simplices, rev=true, by=x->x[2])
#升序排列，观察向下闭包
    list_violating_quadruplet = Vector{Tuple{Vector{Int64}, Float64, Float64}}()
    violation_quadruplet = 0
    quadruplet_count = 0
    violation_quadruplet_negativeterms = 0

    list_violating_triangles = Vector{Tuple{Vector{Int64}, Float64, Float64}}()
    list_simplices_for_filtration = Vector{Pair{Tuple{Int64, Vararg{Int64}}, Float64}}()
    set_simplices = Set()
    edge_counter =0
    counter = 0
    triangles_count = 0
    violation_triangles = 0
    violation_triangles_negativeterms = 0
    list_simplices_all = Vector{Pair{Tuple{Int64, Vararg{Int64}}, Float64}}()
    counter_simplices_all = 0

    for (index, i) in enumerate(sorted_simplices)
        simplices, weight = i

        if length(simplices) <= 2
            push!(list_simplices_all, (Tuple(simplices) => -weight))
            push!(list_simplices_for_filtration, (Tuple(simplices) => -weight))#加入过滤，权重变为距离
            push!(set_simplices, simplices) 
            edge_counter +=1
        elseif length(simplices) == 3
            flag = 0
            for t in combinations(simplices, 2)
                if t in set_simplices
                    flag += 1
                end
            end

            if flag == 3
                push!(set_simplices, simplices)
                push!(list_simplices_for_filtration, (Tuple(simplices) => -weight))
                counter_simplices_all += 1
                
                push!(list_simplices_all, (Tuple(simplices) => -weight))
              

                if weight >= 0
                    triangles_count += 1
                end
            else
                if weight >= 0
                    violation_triangles += 1
                    push!(list_violating_triangles, (simplices, abs(weight), 3 - flag))
                else
                    violation_triangles_negativeterms += 1
                end
            end
        elseif length(simplices) == 4
            flag = 0
            for t in combinations(simplices, 3)
                if t in set_simplices
                    flag += 1
                end
            end

            if flag == 4
                push!(set_simplices, simplices)
                push!(list_simplices_for_filtration, (Tuple(simplices) => -weight))
               
                push!(list_simplices_all, (Tuple(simplices) => -weight))
                counter += 1
                if weight >= 0
                    quadruplet_count += 1
                end
            else
                if weight >= 0
                    violation_quadruplet += 1
                    push!(list_violating_quadruplet, (simplices, abs(weight), 4 - flag))
                else
                    violation_quadruplet_negativeterms += 1
                end
            end
        end
    end
    println(edge_counter)
    println(counter_simplices_all)
    println(counter)
    #total_triangles = triangles_count + violation_triangles
    #hyper_coherence_violation_triangles = total_triangles == 0 ? 0.0 : (1.0 * violation_triangles) / total_triangles

    #total_quadruplets = quadruplet_count + violation_quadruplet
    #hyper_coherence_violation_quadruplet = total_quadruplets == 0 ? 0.0 : (1.0 * violation_quadruplet) / total_quadruplets
    num_regions=simplicial_TS.num_ROI
    p_tensor_scaffold_b = zeros(num_regions,num_regions)
    p_tensor_scaffold_tri = zeros(num_regions,num_regions)
    if counter_simplices_all > 0 
       p_tensor_scaffold_b,p_tensor_scaffold_tri=compute_scaffold(simplicial_TS,list_simplices_for_filtration,list_simplices_all,t_current)
    end
    return (p_tensor_scaffold_b,p_tensor_scaffold_tri)
end
#紧跟上述的两个列表，进行持续同调，返回一个matrix,同源支架
function compute_scaffold(simplicial_TS::simplicial_complex_mvts,  list_simplices_positive::Vector{Pair{Tuple{Int64, Vararg{Int64}}, Float64}},
     list_filtration_scaffold::Vector{Pair{Tuple{Int64, Vararg{Int64}}, Float64}},
      t_current::Int64,)
       num_nodes=simplicial_TS.num_ROI
       #=
       ##From here it starts the computation of hypercomplexity and scaffold
       dgms1_clean_FC=Vector{Tuple{Real,Real}}()
       dgms1_clean_CT=Vector{Tuple{Real,Real}}()
       dgms1_clean_FD=Vector{Tuple{Real,Real}}()
   
       dgms1=ripserer(Custom(list_simplices_positive),reps=1, alg=:cohomology,dim_max=2, verbose=false)
       println(dgms1)
       println("dgms1")
       println("length(dgms1[3]): ", length(dgms1[3]))
      
       max_filtration_weight = find_max_weight(simplicial_TS,t_current)
       dgms1_clean = [(death(i) == Inf ? (birth(i), max_filtration_weight) : (birth(i), death(i))) for i in dgms1[3]]#计算H2

       for i in dgms1[3]
           (a,b)=(birth(i), death(i))
           if b==Inf
               b=max_filtration_weight
           end
           # println(a, " ", b)
       end    
       for i in dgms1[3]
           (a,b)=(birth(i), death(i))
           if b==Inf
               b=max_filtration_weight
           end
           if a<0 
               if b<=0
                   push!(dgms1_clean_FC,(a,b))
               else
                   push!(dgms1_clean_CT,(a,b))
               end
           else
               push!(dgms1_clean_FD,(a,b))
           end
       end
       println("ok")
      
        trivial_diagram=PersistenceDiagram([]; )
           
        complexity_FC = Wasserstein()(PersistenceDiagram(dgms1_clean_FC), trivial_diagram)
        complexity_CT = Wasserstein()(PersistenceDiagram(dgms1_clean_CT), trivial_diagram)
        complexity_FD = Wasserstein()(PersistenceDiagram(dgms1_clean_FD), trivial_diagram)
           #hyper_complexity = complexity_FC+ complexity_CT+complexity_FD
        hyper_complexity = Wasserstein()(PersistenceDiagram(dgms1_clean), trivial_diagram)
       
       
      =#
        dgms1=ripserer(Custom(list_filtration_scaffold),reps= true, alg=:homology,dim_max=2, verbose=false)
        f_tensor_scaffold_b,p_tensor_scaffold_b=compute_scaffold_fast(dgms1,num_nodes)
        f_tensor_scaffold_b2,p_tensor_scaffold_tri=compute_scaffold_fast_tri(dgms1,num_nodes)
        println("dgms1:  ")
        println(dgms1)
       
    return (p_tensor_scaffold_b,p_tensor_scaffold_tri)
    #println(t_current, " ", hyper_complexity, " ",complexity_FC, " ",complexity_CT, " ",complexity_FD, " ",hyper_coherence, " ",avg_edge_violation)
end

function find_max_weight(list_simplices_positive, t)
    edges_abs_max = list_simplices_positive.ets_max[t]
    triplets_abs_max = list_simplices_positive.triplets_max[t]
    m = max(edges_abs_max, triplets_abs_max)
    return(m)
end
#下面这个是根据空腔中的节点转化为一个加权图
function compute_scaffold_fast(diagrams,num_regions)
    p_tensor_scaffold_b = zeros(num_regions,num_regions)
    f_tensor_scaffold_b = zeros(num_regions,num_regions)
    t = 0.0001
    eps = 0.01

    list_persistences_nb = Any[]
    list_persistences_b = Any[]
    list_persistences_c = Any[]
#     list_births_nb = Any[]
#     list_births_b = Any[]
#     list_births_c = Any[]

#     list_deaths_nb = Any[]
#     list_deaths_b = Any[]
#     list_deaths_c = Any[]

#     list_lengths_nb = Any[]
#     list_lengths_b = Any[]
#     list_lengths_c = Any[]

    count = 0
    c_fail = 0
    d = diagrams
    f = d[3].filtration
    persistences = []
    births = []
    deaths = []
    lengths = []
    for l in eachindex(d[3])
        g = d[3][l]
        p = persistence(g)
#         append!(persistences, p)
#         append!(births, birth(g))
#         append!(deaths, death(g))
#         append!(lengths,length(representative(g)))
        
        for rep in representative(g)
            count += 1
            i , j , k  = vertices(rep)
            
            @inbounds p_tensor_scaffold_b[i,j] += p
            @inbounds p_tensor_scaffold_b[i,k] += p
            @inbounds p_tensor_scaffold_b[k,j] += p
            #@inbounds f_tensor_scaffold_b[i,j] += 1
            @inbounds p_tensor_scaffold_b[j,i] += p
            @inbounds p_tensor_scaffold_b[k,i] += p
            @inbounds p_tensor_scaffold_b[j,k] += p
            #@inbounds f_tensor_scaffold_b[j,i] += 1
        end
    end
#     push!(list_persistences_b, persistences)
#     push!(list_births_b, births)
#     push!(list_deaths_b, deaths)
#     push!(list_lengths_b, lengths)
    #return (p_tensor_scaffold_b,f_tensor_scaffold_b)
    return (f_tensor_scaffold_b,p_tensor_scaffold_b)
end
function compute_scaffold_fast_tri(diagrams,num_regions)
    p_tensor_scaffold_b = zeros(num_regions,num_regions)
    f_tensor_scaffold_b = zeros(num_regions,num_regions)
    t = 0.0001
    eps = 0.01

    list_persistences_nb = Any[]
    list_persistences_b = Any[]
    list_persistences_c = Any[]
#     list_births_nb = Any[]
#     list_births_b = Any[]
#     list_births_c = Any[]

#     list_deaths_nb = Any[]
#     list_deaths_b = Any[]
#     list_deaths_c = Any[]

#     list_lengths_nb = Any[]
#     list_lengths_b = Any[]
#     list_lengths_c = Any[]

    count = 0
    c_fail = 0
    d = diagrams
    f = d[2].filtration
    persistences = []
    births = []
    deaths = []
    lengths = []
    for l in eachindex(d[2])
        g = d[2][l]
        p = persistence(g)
#         append!(persistences, p)
#         append!(births, birth(g))
#         append!(deaths, death(g))
#         append!(lengths,length(representative(g)))
        for rep in representative(g)
            count += 1
            i , j = vertices(rep)
            
            @inbounds p_tensor_scaffold_b[i,j] += p
            #@inbounds f_tensor_scaffold_b[i,j] += 1
            @inbounds p_tensor_scaffold_b[j,i] += p
            #@inbounds f_tensor_scaffold_b[j,i] += 1
        end
    end
#     push!(list_persistences_b, persistences)
#     push!(list_births_b, births)
#     push!(list_deaths_b, deaths)
#     push!(list_lengths_b, lengths)
    #return (p_tensor_scaffold_b,f_tensor_scaffold_b)
    return (f_tensor_scaffold_b,p_tensor_scaffold_b)
end
#下面这个函数用于计算加权矩阵
function compute_edgeweight(list_violations)
    edge_weight = Dict{Tuple{Int, Int}, Vector{Float64}}()
    for element in list_violations
        triplets, weight = element
        for edge in combinations(triplets, 2)
            edge_tuple = (edge[1], edge[2])
            if haskey(edge_weight, edge_tuple)
                edge_weight[edge_tuple][1] += weight
                edge_weight[edge_tuple][2] += 1.0
            else
                edge_weight[edge_tuple] = [weight, 1.0]
            end
        end
    end
    return edge_weight
end

function search_good_quadruplet(sorted_simplices::Vector{Tuple{Vector{Int64}, Float64}})::Tuple{Matrix{Float64}, Matrix{Float64}}
    # 按权重升序排序
    sort!(sorted_simplices, rev=false, by=x->x[2])

    # 初始化变量
    list_good_quadruplet = Vector{Tuple{Vector{Int64}, Float64, Float64}}()
    list_good_triangles = Vector{Tuple{Vector{Int64}, Float64, Float64}}()
    set_simplices = Set()
    triangles_count = 0
    good_triangles_count = 0
    quadruplet_count = 0
    good_quadruplet_count = 0

    for (index, i) in enumerate(sorted_simplices)
        simplices, weight = i
        if length(simplices) <= 2
            push!(set_simplices, simplices)
            
        elseif length(simplices) == 3
            flag = 0
            for t in combinations(simplices, 2)
                if t in set_simplices
                    flag += 1
                end
            end
            if weight >= 0
                triangles_count += 1
            end
            if flag == 3
                push!(set_simplices, simplices)

                if weight >= 0
                    good_triangles_count += 1
                    push!(list_good_triangles, (simplices, abs(weight), 3 - flag))
                    
                end
            end
        elseif length(simplices) == 4
            flag = 0
            for t in combinations(simplices, 3)
                if t in set_simplices
                    flag += 1
                end
            end
            if weight >= 0
                quadruplet_count += 1
            end
            if flag == 4
             
                if weight >= 0
                    good_quadruplet_count += 1
                    push!(list_good_quadruplet, (simplices, abs(weight), 4 - flag))
                end
            end
        end
    end
    println("good_quadruplet_count ：")
    println(good_quadruplet_count)
    # 计算比例
  
    # 计算边权重
    triangles_edge_weight = compute_edgeweight(list_good_triangles)
    quadruplet_edge_weight = compute_edgeweight(list_good_quadruplet)

    # 初始化矩阵并填充（注意 Julia 的 1-based 索引）
    triangles_matrix = zeros(90, 90)
    quadruplet_matrix = zeros(90, 90)

    for (edgeID, weights) in triangles_edge_weight
        i, j = edgeID
      
        w = weights[1]
        triangles_matrix[i, j] = w  # 转换到 1-based 索引
        triangles_matrix[j, i] = w
    end

    for (edgeID, weights) in quadruplet_edge_weight
        i, j = edgeID
        w = weights[1]
        quadruplet_matrix[i, j] = w
        quadruplet_matrix[j, i] = w
    end

    return triangles_matrix, quadruplet_matrix
    
end

#上面这个函数用于计算四面体的加权矩阵

function fix_violations_and_compute_complexity(sorted_simplices::Vector{Tuple{Vector{Int64}, Float64}})::Vector{Float64}
    # Sorting the simplices in a descending order according to the weights
    sort!(sorted_simplices, rev=true, by=x->x[2])

    list_violating_quadruplet = Vector{Tuple{Vector{Int64}, Float64, Float64}}()
    violation_quadruplet = 0
    quadruplet_count = 0
    violation_quadruplet_negativeterms = 0

    list_violating_triangles = Vector{Tuple{Vector{Int64}, Float64, Float64}}()
    list_simplices_for_filtration = Vector{Pair{Tuple{Int64, Vararg{Int64}}, Float64}}()
    set_simplices = Set()
    counter = 0
    triangles_count = 0
    violation_triangles = 0
    violation_triangles_negativeterms = 0
    list_simplices_all = Vector{Pair{Tuple{Int64, Vararg{Int64}}, Float64}}()
    counter_simplices_all = 0

    for (index, i) in enumerate(sorted_simplices)
        simplices, weight = i

        if length(simplices) <= 2
            push!(list_simplices_all, (Tuple(simplices) => -weight))
            push!(list_simplices_for_filtration, (Tuple(simplices) => -weight))
            push!(set_simplices, simplices)
            counter += 1
        elseif length(simplices) == 3
            flag = 0
            for t in combinations(simplices, 2)
                if t in set_simplices
                    flag += 1
                end
            end

            if flag == 3
                push!(set_simplices, simplices)
                push!(list_simplices_for_filtration, (Tuple(simplices) => -weight))
                counter += 1
                if weight != sorted_simplices[index - 1][2]
                    counter_simplices_all += 1
                    push!(list_simplices_all, (Tuple(simplices) => -weight))
                else
                    push!(list_simplices_all, (Tuple(simplices) => -weight))
                end

                if weight >= 0
                    triangles_count += 1
                end
            else
                if weight >= 0
                    violation_triangles += 1
                    push!(list_violating_triangles, (simplices, abs(weight), 3 - flag))
                else
                    violation_triangles_negativeterms += 1
                end
            end
        elseif length(simplices) == 4
            flag = 0
            for t in combinations(simplices, 3)
                if t in set_simplices
                    flag += 1
                end
            end

            if flag == 4
                push!(set_simplices, simplices)
                push!(list_simplices_for_filtration, (Tuple(simplices) => -weight))
                counter += 1
                if weight >= 0
                    quadruplet_count += 1
                end
            else
                if weight >= 0
                    violation_quadruplet += 1
                    push!(list_violating_quadruplet, (simplices, abs(weight), 4 - flag))
                else
                    violation_quadruplet_negativeterms += 1
                end
            end
        end
    end

    #total_triangles = triangles_count + violation_triangles
    #hyper_coherence_violation_triangles = total_triangles == 0 ? 0.0 : (1.0 * violation_triangles) / total_triangles

    total_quadruplets = quadruplet_count + violation_quadruplet
    hyper_coherence_violation_quadruplet = total_quadruplets == 0 ? 0.0 : (1.0 * violation_quadruplet) / total_quadruplets

    return  [hyper_coherence_violation_quadruplet]
end

#这个函数帮助产生一个H2的过滤，我要重新定义hyper_coherence_violation_quadruplet
#先考虑只要3阶共波动大于所有的2阶共波动的四面体
function fix_violations_and_compute_qua_coherence(sorted_simplices::Vector{Tuple{Vector{Int64}, Float64}})::Vector{Float64}
    # Sorting the simplices in a 升序 order according to the weights
    sort!(sorted_simplices, rev=false, by=x->x[2])

    
    set_simplices = Set()
    counter = 0
   
    total_quadruplets = 0
    good_quadruplet_count = 0
    for (index, i) in enumerate(sorted_simplices)
        simplices, weight = i

        if length(simplices) <= 2
            
            push!(set_simplices, simplices)
        elseif length(simplices) == 3
            flag = 0
            for t in combinations(simplices, 2)
                if t in set_simplices
                    flag += 1
                end
            end
            if flag ==3
                push!(set_simplices, simplices)
            end
        elseif length(simplices) == 4
            flag = 0
            for t in combinations(simplices, 3)
                if t in set_simplices
                    flag += 1
                end
            end
            if  weight > 0 
                total_quadruplets += 1
            end
            if flag == 4
                if weight > 0
                    good_quadruplet_count += 1
                end
            end
        end
    end

    hyper_coherence_violation_quadruplet = total_quadruplets == 0 ? 0.0 : (1.0 * good_quadruplet_count) / total_quadruplets

    return  [hyper_coherence_violation_quadruplet]
end

end