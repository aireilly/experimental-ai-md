# Deploying {product-title} in a disconnected environment

You can inference large language models with Red Hat AI Inference Server without any connection to the outside internet by installing OpenShift Container Platform and configuring a mirrored container image registry in the disconnected environment.

<dl><dt><strong>❗ IMPORTANT</strong></dt><dd>

Currently, only NVIDIA CUDA AI accelerators are supported for OpenShift Container Platform in disconnected environments.
</dd></dl>

# Setting up a mirror registry for your disconnected environment

To serve container images in a disconnected environment, you must configure a disconnected mirror registry on a bastion host.
The bastion host acts as a secure gateway between your disconnected environment and the internet.
You then mirror images from Red Hat’s online image registries, and serve them in the disconnected environment.

* Deploy the bastion host.
* [Install `oc`](https://docs.redhat.com/en/documentation/openshift_container_platform/latest/html-single/cli_tools/index#installing-openshift-cli) in the bastion host.
* [Install Podman](https://podman.io/docs/installation) in the bastion host.
* [Install OpenShift Container Platform](https://docs.redhat.com/en/documentation/openshift_container_platform/latest/html/disconnected_environments/installing-disconnected-environments) in the disconnected environment.

1. Open a shell prompt on the bastion host and [create the disconnected mirror registry](https://docs.redhat.com/en/documentation/openshift_container_platform/latest/html/disconnected_environments/mirroring-in-disconnected-environments#installing-mirroring-creating-registry).
2. [Configure credentials that allow images to be mirrored](https://docs.redhat.com/en/documentation/openshift_container_platform/latest/html/disconnected_environments/mirroring-in-disconnected-environments#installation-adding-registry-pull-secret_installing-mirroring-disconnected).

# Mirroring the required images for model inference

Once you have created a mirror registry for the disconnected environment, you are ready to mirror the required AI Inference Server image, AI accelerator Operator images, and language model image.

* You have installed the OpenShift CLI (`oc`).
* You have logged in as a user with `cluster-admin` privileges.
* You have installed a mirror registry on the bastion host.

1. Find the version of the following images that match your environment and pull the images with `podman`:
    * [Node Feature Discovery (NFD) Operator](https://catalog.redhat.com/search?gs&q=Node+Feature+Discovery+%28NFD%29+Operator&searchType=all)
    * [NVIDIA GPU Operator](https://catalog.redhat.com/search?gs&q=gpu%20operator)
    * [AI Inference Server](https://catalog.redhat.com/search?gs&q=red%20hat%20ai%20inference%20server)
2. [Create an image set configuration custom resource (CR)](https://docs.redhat.com/en/documentation/openshift_container_platform/latest/html/disconnected_environments/mirroring-in-disconnected-environments#oc-mirror-building-image-set-config-v2_about-installing-oc-mirror-v2) that includes the NFD Operator, NVIDIA GPU Operator, and AI Inference Server images that you pulled in the previous step.
For example, save the following `ImageSetConfiguration` CR as the file `imageset-config.yaml`:

    ```yaml
    apiVersion: mirror.openshift.io/v2alpha1
    kind: ImageSetConfiguration
    mirror:
      operators:
      # Node Feature Discovery (NFD) Operator
      # Helps Openshift detect hardware capabilities like GPUs
      - catalog: registry.redhat.io/openshift4/ose-cluster-nfd-operator:latest
        packages:
          - name: nfd
            defaultChannel: stable
            channels:
              - name: stable

      # GPU Operator
      # Manages NVIDIA GPUs on OpenShift
      - catalog: registry.connect.redhat.com/nvidia/gpu-operator-bundle:latest
        packages:
          - name: gpu-operator-certified
            defaultChannel: stable
            channels:
              - name: stable
      additionalImages:
      # Red Hat AI Inference Server image
      - name: registry.redhat.io/rhaiis/vllm-cuda-rhel9:latest
      # Model image
      - name: registry.redhat.io/rhelai1/granite-3-1-8b-instruct-quantized-w8a8:1.5
    ```
3. Mirror the required images into the mirror registry. Run the following command:

    ```terminal
    $ oc mirror --config imageset-config.yaml docker://<target_mirror_registry_url> --registry-config <path_to_pull_secret_json>
    ```
4. Alternatively, if you have already installed the NFD and NVIDIA GPU Operators in the cluster, create an `ImageSetConfiguration` CR that configures AI Inference Server and model images only:

    ```yaml
    apiVersion: mirror.openshift.io/v2alpha1
    kind: ImageSetConfiguration
    mirror:
      additionalImages:
      - name: registry.redhat.io/rhaiis/vllm-cuda-rhel9:latest
      - registry.redhat.io/rhelai1/granite-3-1-8b-instruct-quantized-w8a8:1.5
    ```
5. [Mirror the image set in the disconnected environment](https://docs.redhat.com/en/documentation/openshift_container_platform/latest/html/disconnected_environments/mirroring-in-disconnected-environments#oc-mirror-workflows-partially-disconnected-v2_about-installing-oc-mirror-v2).
6. [Configure the disconnected cluster to use the updated image set](https://docs.redhat.com/en/documentation/openshift_container_platform/latest/html/disconnected_environments/mirroring-in-disconnected-environments#oc-mirror-updating-cluster-manifests-v2_about-installing-oc-mirror-v2).

# Installing the Node Feature Discovery Operator

Install the Node Feature Discovery Operator so that the cluster can use the AI accelerators that are available in the cluster.

* You have installed the OpenShift CLI (`oc`).
* You have logged in as a user with `cluster-admin` privileges.

1. Create the `Namespace` CR for the Node Feature Discovery Operator:

    ```yaml
    oc apply -f - <<EOF
    apiVersion: v1
    kind: Namespace
    metadata:
      name: openshift-nfd
      labels:
        name: openshift-nfd
        openshift.io/cluster-monitoring: "true"
    EOF
    ```
2. Create the `OperatorGroup` CR:

    ```yaml
    oc apply -f - <<EOF
    apiVersion: operators.coreos.com/v1
    kind: OperatorGroup
    metadata:
      generateName: openshift-nfd-
      name: openshift-nfd
      namespace: openshift-nfd
    spec:
      targetNamespaces:
      - openshift-nfd
    EOF
    ```
3. Create the `Subscription` CR:

    ```yaml
    oc apply -f - <<EOF
    apiVersion: operators.coreos.com/v1alpha1
    kind: Subscription
    metadata:
      name: nfd
      namespace: openshift-nfd
    spec:
      channel: "stable"
      installPlanApproval: Automatic
      name: nfd
      source: redhat-operators
      sourceNamespace: openshift-marketplace
    EOF
    ```

Verify that the Node Feature Discovery Operator deployment is successful by running the following command:

```terminal
$ oc get pods -n openshift-nfd
```

**Example output**

```terminal
NAME                                      READY   STATUS    RESTARTS   AGE
nfd-controller-manager-7f86ccfb58-vgr4x   2/2     Running   0          10m
```

* [Installing the Node Feature Discovery Operator](https://docs.redhat.com/en/documentation/openshift_container_platform/latest/html/specialized_hardware_and_driver_enablement/psap-node-feature-discovery-operator#installing-the-node-feature-discovery-operator_psap-node-feature-discovery-operator)

# Installing the NVIDIA GPU Operator

Install the NVIDIA GPU Operator to use the underlying NVIDIA CUDA AI accelerators that are available in the cluster.

* You have installed the OpenShift CLI (`oc`).
* You have logged in as a user with `cluster-admin` privileges.
* You have installed the Node Feature Discovery Operator.

1. Create the `Namespace` CR for the NVIDIA GPU Operator:

    ```yaml
    oc apply -f - <<EOF
    apiVersion: v1
    kind: Namespace
    metadata:
      name: nvidia-gpu-operator
    EOF
    ```
2. Create the `OperatorGroup` CR:

    ```yaml
    oc apply -f - <<EOF
    apiVersion: operators.coreos.com/v1
    kind: OperatorGroup
    metadata:
      name: gpu-operator-certified
      namespace: nvidia-gpu-operator
    spec:
     targetNamespaces:
     - nvidia-gpu-operator
    EOF
    ```
3. Create the `Subscription` CR:

    ```yaml
    oc apply -f - <<EOF
    apiVersion: operators.coreos.com/v1alpha1
    kind: Subscription
    metadata:
      name: gpu-operator-certified
      namespace: nvidia-gpu-operator
    spec:
      channel: "stable"
      installPlanApproval: Manual
      name: gpu-operator-certified
      source: certified-operators
      sourceNamespace: openshift-marketplace
    EOF
    ```

Verify that the NVIDIA GPU Operator deployment is successful by running the following command:

```terminal
$ oc get pods -n nvidia-gpu-operator
```

**Example output**

```terminal
NAME                                                  READY   STATUS     RESTARTS   AGE
gpu-feature-discovery-c2rfm                           1/1     Running    0          6m28s
gpu-operator-84b7f5bcb9-vqds7                         1/1     Running    0          39m
nvidia-container-toolkit-daemonset-pgcrf              1/1     Running    0          6m28s
nvidia-cuda-validator-p8gv2                           0/1     Completed  0          99s
nvidia-dcgm-exporter-kv6k8                            1/1     Running    0          6m28s
nvidia-dcgm-tpsps                                     1/1     Running    0          6m28s
nvidia-device-plugin-daemonset-gbn55                  1/1     Running    0          6m28s
nvidia-device-plugin-validator-z7ltr                  0/1     Completed  0          82s
nvidia-driver-daemonset-410.84.202203290245-0-xxgdv   2/2     Running    0          6m28s
nvidia-node-status-exporter-snmsm                     1/1     Running    0          6m28s
nvidia-operator-validator-6pfk6                       1/1     Running    0          6m28s
```

* [Installing the NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/openshift/latest/mirror-gpu-ocp-disconnected.html#deploy-gpu-operators-in-a-disconnected-or-airgapped-environment)

# Inference serving the model in the disconnected environment

Use Red Hat AI Inference Server deployed in a disconnected OpenShift Container Platform environment to inference serve the language model from cluster persistent storage.

* You have installed a mirror registry on the bastion host that is accessible to the disconnected cluster.
* You have added the model and Red Hat AI Inference Server images to the mirror registry.
* You have installed the Node Feature Discovery Operator and NVIDIA GPU Operator in the disconnected cluster.

1. In the disconnected cluster, configure persistent storage using Network File System (NFS) and make the model available in the persistent storage that you configure.

    <dl><dt><strong>📌 NOTE</strong></dt><dd>

    For more information, see [Persistent storage using NFS](https://docs.redhat.com/en/documentation/openshift_container_platform/latest/html/storage/configuring-persistent-storage#persistent-storage-using-nfs).
    </dd></dl>
2. Create a `Deployment` custom resource (CR).
For example, the following `Deployment` CR uses AI Inference Server to serve a Granite model on a CUDA accelerator.

    ```yaml
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: granite
      namespace: rhaiis-namespace
      labels:
        app: granite
    spec:
      replicas: 0
      selector:
        matchLabels:
          app: granite
      template:
        metadata:
          labels:
            app: granite
        spec:
          containers:
            - name: granite
              image: 'registry.redhat.io/rhaiis/vllm-cuda-rhel9@sha256:137ac606b87679c90658985ef1fc9a26a97bb11f622b988fe5125f33e6f35d78'
              imagePullPolicy: IfNotPresent
              command:
                - python
                - '-m'
                - vllm.entrypoints.openai.api_server
              args:
                - '--port=8000'
                - '--model=/mnt/models' ①
                - '--served-model-name=granite-3.1-2b-instruct-quantized.w8a8'
                - '--tensor-parallel-size=1'
              resources:
                limits:
                  cpu: '10'
                  nvidia.com/gpu: '1'
                requests:
                  cpu: '2'
                  memory: 6Gi
                  nvidia.com/gpu: '1'
              volumeMounts:
                - name: cache-volume
                  mountPath: /mnt/models
                - name: shm
                  mountPath: /dev/shm ②
          volumes:
            - name: cache-volume
              persistentVolumeClaim:
                claimName: granite-31-w8a8
            - name: shm
              emptyDir:
                medium: Memory
                sizeLimit: 2Gi
          restartPolicy: Always
    ```
    1. The model that you downloaded should be available from this mounted location in the configured persistent volume.
    2. The `/dev/shm` volume mount is required by the NVIDIA Collective Communications Library (NCCL).
    Tensor parallel vLLM deployments fail when the `/dev/shm` volume mount is not set.
3. Create a `Service` CR for the model inference. For example:

    ```yaml
    apiVersion: v1
    kind: Service
    metadata:
      name: granite
      namespace: rhaiis-namespace
    spec:
      selector:
        app: granite
      ports:
        - protocol: TCP
          port: 80
          targetPort: 8000
    ```
4. Optional. Create a `Route` CR to enable public access to the model. For example:

    ```yaml
    apiVersion: route.openshift.io/v1
    kind: Route
    metadata:
      name: granite
      namespace: rhaiis-namespace
    spec:
      to:
        kind: Service
        name: granite
      port:
        targetPort: 80
    ```
5. Get the URL for the exposed route:

    ```terminal
    $ oc get route granite -n rhaiis-namespace -o jsonpath='{.spec.host}'
    ```

    **Example output**

    ```terminal
    granite-rhaiis-namespace.apps.example.com
    ```
6. Query the model by running the following command:

    ```terminal
    curl -X POST http://granite-rhaiis-namespace.apps.example.com/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "granite-3.1-2b-instruct-quantized.w8a8",
        "messages": [{"role": "user", "content": "What is AI?"}],
        "temperature": 0.1
      }'
    ```

* [Understanding deployments](https://docs.redhat.com/en/documentation/openshift_container_platform/latest/html/building_applications/deployments#what-deployments-are)
