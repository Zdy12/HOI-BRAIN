#这个函数用于计算闭包四面体和闭包三角形的加权矩阵.

#用来产生时间导数的矩阵，和四面体的空腔的加权矩阵。
using FilePathsBase
using DelimitedFiles
push!(LOAD_PATH, "C:\\Users\\sd\\Desktop\\期刊的hoi最新/")
using SimplicialTS_NEW
using Random
using DataFrames
using DataFrames: select, Not  # 明确导入 DataFrame 的函数
using LinearAlgebra
using MAT
println("=======================")
# 读取单个 .mat 文件并返回其数据矩阵
function load_data_mat(path_single_file::String)::Matrix{Float64}
    file_to_open = path_single_file
    data = matread(file_to_open)
    key_data = collect(keys(data))[end]
    data = data[key_data]
    data = data[1:80,1:90]
    return data
end

# 遍历文件夹中的所有 .mat 文件，并将它们的数据合并成一个三维数组
function merge_mat_files(folder::String)::Array{Float64, 3}
    mat_files = String[]
    for (root, dirs, files) in walkdir(folder)
        for file in files
            if endswith(file, ".mat") && occursin("AAL116_features_timeseries", file)
                push!(mat_files, joinpath(root, file))
            end
        end
    end

    # 检查是否有找到任何 .mat 文件
    if isempty(mat_files)
        error("No .mat files found in the specified folder")
    end

    # 初始化三维数组的维度
    # 假设所有文件的数据维度相同
    sample_data = load_data_mat(mat_files[1])
    num_rois, num_timepoints = size(sample_data')
    num_files = length(mat_files)

    # 创建一个三维数组来存储所有文件的数据，维度为：病人 × 脑区 × 时间序列
    merged_data = zeros(Float64, num_files, num_rois, num_timepoints)

    # 遍历所有 .mat 文件并填充三维数组
    for (i, mat_file) in enumerate(mat_files)
        data = load_data_mat(mat_file)
        merged_data[i, :, :] = data'  # 按病人索引填充数据
    end

    return merged_data
end


function extract_features(processed_data, T)
    num_samples = size(processed_data, 1)
    features_edge_positive = Matrix{Float64}(undef, num_samples, 8100)
    features_edge_negative = Matrix{Float64}(undef, num_samples, 8100)

    features_tri_positive = Matrix{Float64}(undef, num_samples, 8100)
    features_qua_positive = Matrix{Float64}(undef, num_samples, 8100)

    features_tri_negative = Matrix{Float64}(undef, num_samples, 8100)
    features_qua_negative = Matrix{Float64}(undef, num_samples, 8100)


    for i in 1:num_samples
        sum_output_edge_positive = zeros(90, 90)
        sum_output_edge_negative = zeros(90, 90)

        sum_output_tri_positive = zeros(90, 90)
        sum_output_qua_positive = zeros(90, 90)

        sum_output_tri_negative = zeros(90, 90)
        sum_output_qua_negative = zeros(90, 90)
        simplicial_TS = create_data_structure(processed_data[i, :, :])
        println(i)
        println()
        for t in 1:T-1
            # 获取当前时间点的数据
            list_all_simplices_positive, list_all_simplices_negative,edges_matrix_positive,edges_matrix_negative = create_simplicial_complex(simplicial_TS, t)
        
            sum_output_edge_positive .+= edges_matrix_positive
            sum_output_edge_negative .+= edges_matrix_negative
            
            print("       ")
            println(t)
            
            # 计算特征向量
            triangle_matrix_positive, quadruplet_matrix_positive = search_good_quadruplet(list_all_simplices_positive)

            triangle_matrix_negative, quadruplet_matrix_negative = search_good_quadruplet(list_all_simplices_negative)

            sum_output_tri_positive .+= triangle_matrix_positive
            sum_output_qua_positive .+= quadruplet_matrix_positive

            sum_output_tri_negative .+= triangle_matrix_negative
            sum_output_qua_negative .+= quadruplet_matrix_negative
           
        end
        avg_output_edge_positive = sum_output_edge_positive / (T - 1)
        avg_output_edge_negative = sum_output_edge_negative / (T - 1)
        features_edge_positive[i, :] = avg_output_edge_positive[:]
        features_edge_negative[i, :] = avg_output_edge_negative[:]

        avg_output_tri_positive = sum_output_tri_positive / (T - 1)
        avg_output_qua_positive = sum_output_qua_positive / (T - 1)
        features_tri_positive[i, :] = avg_output_tri_positive[:]
        features_qua_positive[i, :] = avg_output_qua_positive[:]

        avg_output_tri_negative = sum_output_tri_negative / (T - 1)
        avg_output_qua_negative = sum_output_qua_negative / (T - 1)
        features_tri_negative[i, :] = avg_output_tri_negative[:]
        features_qua_negative[i, :] = avg_output_qua_negative[:]

    end
    return features_tri_positive,features_qua_positive,features_tri_negative,features_qua_negative
end

Random.seed!(42)
# Path handling
desktop_path1 = joinpath(homedir(), "Desktop", "ppmi_new", "ppmi_new","patients")
merged_data1 = merge_mat_files(desktop_path1)
println(size(merged_data1)) 

desktop_path2 = joinpath(homedir(), "Desktop", "ppmi_new", "ppmi_new","prodromal")
merged_data2 = merge_mat_files(desktop_path2)
println(size(merged_data2)) 



T = 70

println("=======================")
#t=1
#simplicial_TS = create_data_structure(processed_data_AD[1, :, :])
#list_all_simplices = create_simplicial_complex(simplicial_TS, t)
#println("List of all simplices:")
#output=fix_violations_and_compute_scaffold(list_all_simplices, t, simplicial_TS)
#println(join(output, ' '))
#println(size(output))
PPMI_paitents_features_tri_positive,PPMI_paitents_features_qua_positive,PPMI_paitents_features_tri_negative,PPMI_paitents_features_qua_negative = extract_features(merged_data1, T)
println("=======================AD is ok")
PPMI_prodromal_features_tri_positive,PPMI_prodromal_features_qua_positive,PPMI_prodromal_features_tri_negative,PPMI_prodromal_features_qua_negative = extract_features(merged_data2, T)
println("=======================")

# 保存特征到文本文件
using DelimitedFiles

# 保存 features_AD

output_file_ad3 = "PPMI_paitents_features_tri_positive.txt"
writedlm(output_file_ad3, PPMI_paitents_features_tri_positive, '\t')  # 使用制表符分隔

output_file_ad4 = "PPMI_paitents_features_qua_positive.txt"
writedlm(output_file_ad4, PPMI_paitents_features_qua_positive, '\t')  # 使用制表符分隔

output_file_ad5 = "PPMI_paitents_features_tri_negative.txt"
writedlm(output_file_ad5, PPMI_paitents_features_tri_negative, '\t')  # 使用制表符分隔

output_file_ad6 = "PPMI_paitents_features_qua_negative.txt"
writedlm(output_file_ad6, PPMI_paitents_features_qua_negative, '\t')  # 使用制表符分隔

# 保存 features_CN

output_file_CN3 = "PPMI_prodromal_features_tri_positive.txt"
writedlm(output_file_CN3, PPMI_prodromal_features_tri_positive, '\t')  # 使用制表符分隔

output_file_CN4 = "PPMI_prodromal_features_qua_positive.txt"
writedlm(output_file_CN4, PPMI_prodromal_features_qua_positive, '\t')  # 使用制表符分隔

output_file_CN5 = "PPMI_prodromal_features_tri_negative.txt"
writedlm(output_file_CN5, PPMI_prodromal_features_tri_negative, '\t')  # 使用制表符分隔

output_file_CN6 = "PPMI_prodromal_features_qua_negative.txt"
writedlm(output_file_CN6, PPMI_prodromal_features_qua_negative, '\t')  # 使用制表符分隔

