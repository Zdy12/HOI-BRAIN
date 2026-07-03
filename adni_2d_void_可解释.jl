#用来产生时间导数的矩阵，和四面体的空腔的加权矩阵。
using FilePathsBase
using DelimitedFiles
push!(LOAD_PATH, "C:\\Users\\sd\\Desktop\\期刊的hoi最新/")
using SimplicialTS_NEW_interper
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
    
    features_positive = Matrix{Float64}(undef, num_samples, 69)
    features_negative = Matrix{Float64}(undef, num_samples, 69)
   

    for i in 1:num_samples
        positive_number_list = zeros(69)
        negative_number_list = zeros(69)
        simplicial_TS = create_data_structure(processed_data[i, :, :])
        println(i)
        println()
        for t in 1:T-1
            # 获取当前时间点的数据
            list_all_simplices_positive, list_all_simplices_negative,edges_matrix_positive,edges_matrix_negative = create_simplicial_complex(simplicial_TS, t)
        
           
            
            print("       ")
            println(t)
            
            # 计算特征向量
            numberer_positive = fix_violations_and_compute_scaffold(list_all_simplices_positive, t, simplicial_TS)

            numberer_negative = fix_violations_and_compute_scaffold(list_all_simplices_negative, t, simplicial_TS)

            positive_number_list[t] = numberer_positive
            negative_number_list[t] = numberer_negative

            
           
        end
        
        features_positive[i, :] = positive_number_list[:]
        features_negative[i, :] = negative_number_list[:]
        
        
    end
    return features_positive,features_negative
end



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
println(size(processed_data_AD))
println(size(processed_data_CN))
println(size(processed_data_MCI))
#t=1
#simplicial_TS = create_data_structure(processed_data_AD[1, :, :])
#list_all_simplices = create_simplicial_complex(simplicial_TS, t)
#println("List of all simplices:")
#output=fix_violations_and_compute_scaffold(list_all_simplices, t, simplicial_TS)
#println(join(output, ' '))
#println(size(output))
#AD_features_positive,AD_features_negative = extract_features(processed_data_AD, T)
println("=======================AD is ok")
#CN_features_positive,CN_features_negative = extract_features(processed_data_CN, T)
println("=======================")
MCI_features_positive,MCI_features_negative= extract_features(processed_data_MCI, T)
println("=======================")

# 保存特征到文本文件
using DelimitedFiles

# 保存 features_AD
#output_file_ad1 = "AD_features_void_positive.txt"
#writedlm(output_file_ad1, AD_features_positive, '\t')  # 使用制表符分隔

#output_file_ad2 = "AD_features_void_negative.txt"
#writedlm(output_file_ad2, AD_features_negative, '\t')  # 使用制表符分隔

# 保存 features_CN
#output_file_CN1 = "CN_features_void_positive.txt"
#writedlm(output_file_CN1, CN_features_positive, '\t')  # 使用制表符分隔

#output_file_CN2 = "CN_features_void_negative.txt"
#writedlm(output_file_CN2, CN_features_negative, '\t')  # 使用制表符分隔


# 保存 features_MCI
output_file_MCI1 = "MCI_features_void_positive.txt"
writedlm(output_file_MCI1, MCI_features_positive, '\t')  # 使用制表符分隔

output_file_MCI2 = "MCI_features_void_negative.txt"
writedlm(output_file_MCI2, MCI_features_negative, '\t')  # 使用制表符分隔
