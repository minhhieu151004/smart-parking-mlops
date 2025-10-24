// Tên job Jenkins: parking-service-cicd
// Trigger: Webhook từ GitHub khi có 'git push'
// Mục đích: Tự động hóa việc build và deploy application service khi có thay đổi về code.

pipeline {
    agent any

    environment {        
        DOCKER_IMAGE          = "mhiuuu/parking-model-service" 
        
        // ID của credentials Docker Hub bạn đã tạo trong Jenkins
        DOCKER_CREDENTIALS_ID = "dockerhub-credentials" 
        
        // ID của credentials SSH Private Key bạn đã tạo trong Jenkins để truy cập máy chủ deploy
        SSH_CREDENTIALS_ID    = "vm-ssh-key"
        
        // Thông tin đăng nhập vào máy chủ deploy: user@ip_address
        DEPLOY_SERVER         = "your-server-user@your-server-ip" 
    }

    stages {
        // Giai đoạn 1: Lấy mã nguồn mới nhất từ GitHub
        stage('Checkout Code') {
            steps {
                echo '--> Bắt đầu lấy mã nguồn từ Git...'
                git branch 'main', url: 'https://github.com/minhhieu151004/smart-parking-mlops.git'
                echo '--> Lấy mã nguồn thành công.'
            }
        }

        // Giai đoạn 2: Build Docker image từ Dockerfile
        stage('Build Docker Image') {
            steps {
                script {
                    // Tạo một tag duy nhất cho mỗi lần build để dễ dàng quản lý phiên bản
                    def dockerTag = "build-${env.BUILD_NUMBER}"
                    echo "--> Bắt đầu build Docker image: ${DOCKER_IMAGE}:${dockerTag}"
                    
                    // Lệnh 'sh' thực thi một lệnh shell.
                    // Lệnh này build image từ Dockerfile trong thư mục 'model_service/'.
                    sh "docker build -t ${DOCKER_IMAGE}:${dockerTag} -f model_service/Dockerfile ."
                    
                    // Gắn thêm tag 'latest' vào image vừa build để dễ dàng deploy
                    sh "docker tag ${DOCKER_IMAGE}:${dockerTag} ${DOCKER_IMAGE}:latest"
                    echo "--> Build image thành công."
                }
            }
        }

        // Giai đoạn 3: Đẩy image vừa build lên Docker Hub (hoặc registry khác)
        stage('Push to Docker Registry') {
            steps {
                echo "--> Đẩy image ${DOCKER_IMAGE}:latest lên Docker Hub..."
                withCredentials([usernamePassword(credentialsId: DOCKER_CREDENTIALS_ID, usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')]) {
                    // Đăng nhập vào Docker Hub
                    sh "echo ${DOCKER_PASS} | docker login -u ${DOCKER_USER} --password-stdin"
                    // Đẩy image với tag 'latest'
                    sh "docker push ${DOCKER_IMAGE}:latest"
                    // Đẩy image với tag là số build (để lưu trữ lịch sử)
                    sh "docker push ${DOCKER_IMAGE}:${env.BUILD_NUMBER}"
                }
                echo "--> Đẩy image thành công."
            }
        }

        // Giai đoạn 4: Deploy container mới lên máy chủ
        stage('Deploy New Container') {
            steps {
                 // Sử dụng credentials SSH để thực thi lệnh trên máy chủ từ xa
                 withCredentials([sshUserPrivateKey(credentialsId: SSH_CREDENTIALS_ID, keyFileVariable: 'SSH_KEY')]) {
                    echo "--> Bắt đầu deploy container mới tới server ${DEPLOY_SERVER}..."
                    // Khối 'sh' với '<< ENDSSH' cho phép chạy một loạt lệnh trên server từ xa.
                    sh """
                        ssh -i \${SSH_KEY} -o StrictHostKeyChecking=no ${DEPLOY_SERVER} << 'ENDSSH'
                        
                        echo "1. Đang kéo (pull) image mới nhất từ Docker Hub..."
                        docker pull ${DOCKER_IMAGE}:latest

                        echo "2. Đang dừng và xóa container cũ (nếu có)..."
                        docker stop model-service || true
                        docker rm model-service || true

                        echo "3. Đang khởi động container mới..."
                        docker run -d -p 5000:5000 --name model-service --network my-network \\
                            -e MINIO_ENDPOINT=http://minio:9000 \\
                            -e MINIO_ACCESS_KEY=admin \\
                            -e MINIO_SECRET_KEY=password \\
                            -e S3_BUCKET=my-bucket \\
                            ${DOCKER_IMAGE}:latest
                        
                        echo "4. Deploy hoàn tất!"
                        ENDSSH
                    """
                }
            }
        }
    }
    // 'post' là khối lệnh sẽ được thực thi sau khi tất cả các 'stages' đã chạy.
    post {
        // 'always' đảm bảo rằng các lệnh trong khối này luôn được chạy, dù pipeline thành công hay thất bại.
        always {
            echo '--> Dọn dẹp môi trường...'
            // Đăng xuất khỏi Docker Hub để đảm bảo an toàn.
            sh "docker logout"
        }
    }
}

