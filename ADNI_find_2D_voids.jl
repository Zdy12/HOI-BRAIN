#用来产生时间导数的矩阵，和四面体的空腔的加权矩阵。
using FilePathsBase
using DelimitedFiles
push!(LOAD_PATH, "C:\\Users\\sd\\Desktop\\期刊的hoi最新/")
using SimplicialTS_NEW
using Random
using DataFrames
using DataFrames: select, Not  # 明确导入 DataFrame 的函数
using LinearAlgebra
println("=======================")
function merge_bitext(folder)
    bitext_files = String[]
    for (root, dirs, files) in walkdir(folder)
        for file in files
            push!(bitext_files, joinpath(root, file))
        end
    end

    f_result = Vector{Vector{Vector{Float64}}}()
    for bitext_file in bitext_files
        result = Vector{Vector{Float64}}()
        open(bitext_file, "r") do file
            for line in eachline(file)
                stripped_line = strip(line)
                elements = split(stripped_line, '\t')
                row = parse.(Float64, elements)
                push!(result, row)
            end
        end
        push!(f_result, result)
    end
    return f_result, bitext_files
end

function process_data(data, fixed_length)
    processed_data = []
    for patient_data in data
        # Convert to 2D array
        num_rows = length(patient_data)
        num_cols = length(patient_data[1])
        patient_matrix = Matrix{Float64}(undef, num_rows, num_cols)
        for i in 1:num_rows
            patient_matrix[i, :] = Float64.(patient_data[i])
        end

        # Check dimensions
        if ndims(patient_matrix) != 2
            error("Expected 2D array, got $(ndims(patient_matrix))D array")
        end
        #Padding/Truncating
        current_rows, current_cols = size(patient_matrix)
        if current_rows < fixed_length
            padding = zeros(Float64, fixed_length - current_rows, current_cols)
            patient_matrix = vcat(patient_matrix, padding)
        elseif current_rows > fixed_length
            patient_matrix = patient_matrix[1:fixed_length, :]
        end

        # Transpose and store
        transposed = permutedims(patient_matrix)
        push!(processed_data, transposed)
    end

    # Stack along new dimension
    stacked = cat([p for p in processed_data]..., dims=3)
    final_data = permutedims(stacked, (3, 1, 2))
    return final_data
end

function extract_features(processed_data, T)
    num_samples = size(processed_data, 1)
    
    
    features_qua_positive = Matrix{Float64}(undef, num_samples, 8100)
    features_tri_positive = Matrix{Float64}(undef, num_samples, 8100)
    features_qua_negative = Matrix{Float64}(undef, num_samples, 8100)
    features_tri_negative = Matrix{Float64}(undef, num_samples, 8100)
    for i in 1:num_samples
        sum_output_qua_positive = zeros(90, 90)
        sum_output_tri_positive = zeros(90, 90)
        sum_output_qua_negative = zeros(90, 90)
        sum_output_tri_negative = zeros(90, 90)
        simplicial_TS = create_data_structure(processed_data[i, :, :])
        println(i)
        println()
        for t in 1:T-1
            # 获取当前时间点的数据
           list_all_simplices_positive, list_all_simplices_negative,edges_matrix_positive,edges_matrix_negative = create_simplicial_complex(simplicial_TS, t)
            print("       ")
            println(t)
            # 计算特征向量
            output1_qua_positive, output2_tri_positive = fix_violations_and_compute_scaffold(list_all_simplices_positive, t, simplicial_TS)
            output1_qua_negative, output2_tri_negative = fix_violations_and_compute_scaffold(list_all_simplices_negative, t, simplicial_TS)
            
            sum_output_qua_positive .+=  output1_qua_positive
            sum_output_tri_positive .+=  output2_tri_positive
            sum_output_qua_negative .+=  output1_qua_negative
            sum_output_tri_negative .+=  output2_tri_negative
            
        end
        avg_output_qua_positive = sum_output_qua_positive / (T - 1)
        avg_output_tri_positive = sum_output_tri_positive / (T - 1)
        avg_output_qua_negative = sum_output_qua_negative / (T - 1)
        avg_output_tri_negative = sum_output_tri_negative / (T - 1)

        features_qua_positive[i, :] = avg_output_qua_positive[:]
        features_tri_positive[i, :] = avg_output_tri_positive[:]
        features_qua_negative[i, :] = avg_output_qua_negative[:]
        features_tri_negative[i, :] = avg_output_tri_negative[:]
    end
    return features_qua_positive,features_tri_positive,features_qua_negative,features_tri_negative
end

# 主流程
# 设置随机种子保证可重复性
Random.seed!(42)
# Path handling
desktop_path1 = joinpath(homedir(), "Desktop", "TimeHOIAD", "AD")
desktop_path2 = joinpath(homedir(), "Desktop", "TimeHOIAD", "CN")
desktop_path3 = joinpath(homedir(), "Desktop", "TimeHOIAD", "MCI")


# Process data
f_result1_AD, _ = merge_bitext(desktop_path1)
f_result1_CN, _ = merge_bitext(desktop_path2)
f_result1_MCI, _ = merge_bitext(desktop_path3)
T = 70
processed_data_AD = process_data(f_result1_AD, T)
processed_data_CN = process_data(f_result1_CN, T)
processed_data_MCI = process_data(f_result1_MCI, T)

println("=======================")
println(size(processed_data_MCI))

#t=1
#simplicial_TS = create_data_structure(processed_data_AD[1, :, :])
#list_all_simplices = create_simplicial_complex(simplicial_TS, t)
#println("List of all simplices:")
#output=fix_violations_and_compute_scaffold(list_all_simplices, t, simplicial_TS)
#println(join(output, ' '))
#println(size(output))
AD_void_features_qua_positive,AD_cycle_features_tri_positive,AD_void_features_qua_negative,AD_cycle_features_tri_negative = extract_features(processed_data_AD, T)
CN_void_features_qua_positive,CN_cycle_features_tri_positive,CN_void_features_qua_negative,CN_cycle_features_tri_negative = extract_features(processed_data_CN, T)
MCI_void_features_qua_positive,MCI_cycle_features_tri_positive,MCI_void_features_qua_negative,MCI_cycle_features_tri_negative = extract_features(processed_data_MCI, T)




# 保存特征到文本文件
using DelimitedFiles

output_file_AD1 = "AD_void_features_qua_positive.txt"
writedlm(output_file_AD1, AD_void_features_qua_positive, '\t')  # 使用制表符分隔

output_file_AD2 = "AD_cycle_features_tri_positive.txt"
writedlm(output_file_AD2, AD_cycle_features_tri_positive, '\t')  # 使用制表符分隔

output_file_AD3 = "AD_void_features_qua_negative.txt"
writedlm(output_file_AD3, AD_void_features_qua_negative, '\t')  # 使用制表符分隔

output_file_AD4 = "AD_cycle_features_tri_negative.txt"
writedlm(output_file_AD4, AD_cycle_features_tri_negative, '\t')  # 使用制表符分隔

output_file_CN1 = "CN_void_features_qua_positive.txt"
writedlm(output_file_CN1, CN_void_features_qua_positive, '\t')  # 使用制表符分隔

output_file_CN2 = "CN_cycle_features_tri_positive.txt"
writedlm(output_file_CN2, CN_cycle_features_tri_positive, '\t')  # 使用制表符分隔

output_file_CN3 = "CN_void_features_qua_negative.txt"
writedlm(output_file_CN3, CN_void_features_qua_negative, '\t')  # 使用制表符分隔

output_file_CN4 = "CN_cycle_features_tri_negative.txt"
writedlm(output_file_CN4, CN_cycle_features_tri_negative, '\t')  # 使用制表符分隔


output_file_MCI1 = "MCI_void_features_qua_positive.txt"
writedlm(output_file_MCI1, MCI_void_features_qua_positive, '\t')  # 使用制表符分隔

output_file_MCI2 = "MCI_cycle_features_tri_positive.txt"
writedlm(output_file_MCI2, MCI_cycle_features_tri_positive, '\t')  # 使用制表符分隔

output_file_MCI3 = "MCI_void_features_qua_negative.txt"
writedlm(output_file_MCI3, MCI_void_features_qua_negative, '\t')  # 使用制表符分隔

output_file_MCI4 = "MCI_cycle_features_tri_negative.txt"
writedlm(output_file_MCI4, MCI_cycle_features_tri_negative, '\t')  # 使用制表符分隔