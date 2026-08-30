// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "oaitt-swift",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "GigaAM", targets: ["GigaAM"]),
        .executable(name: "oaitt-swift", targets: ["oaitt-swift"]),
        .executable(name: "OAITT", targets: ["OAITT"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.31.6"),
        .package(url: "https://github.com/hummingbird-project/hummingbird.git", from: "2.26.0"),
        .package(url: "https://github.com/apple/swift-nio-extras.git", from: "1.20.0"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.5.0"),
    ],
    targets: [
        .target(
            name: "GigaAM",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFFT", package: "mlx-swift"),
            ]
        ),
        .executableTarget(
            name: "OAITT",
            dependencies: ["GigaAM"]
        ),
        .executableTarget(
            name: "oaitt-swift",
            dependencies: [
                "GigaAM",
                .product(name: "Hummingbird", package: "hummingbird"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "NIOHTTPTypes", package: "swift-nio-extras"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
    ]
)
