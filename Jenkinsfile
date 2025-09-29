pipeline {
    agent any

    environment {
        CONDA_ENV = "ipolytics"
        PROJECT_ROOT = "${WORKSPACE}"
    }

    triggers {
        cron('H 0 * * 0')
    }

    stages {

        stage('Checkout Code') {
            steps {
                echo "Checking out the repository..."
                checkout scm
            }
        }

        stage('Check MongoDB for New Data') {
            steps {
                echo "Checking for new data in MongoDB..."
                script {

                    def status = bat(
                        script: "conda activate ${CONDA_ENV} && python src\\mlops\\jenkins\\check_new_data.py",
                        returnStatus: true
                    )
                    if (status != 0) {
                        error("No new data. Pipeline stopped.")
                    }
                }
            }
        }

        stage('Run ML Pipelines') {
            steps {
                echo "New data found! Running main.py..."
                bat "conda activate ${CONDA_ENV} && python main.py"
            }
        }
    }

    post {
        success {
            echo ' Pipeline executed successfully!'
        }
        failure {
            echo ' Pipeline failed or no new data.'
        }
    }
}
